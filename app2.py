import nest_asyncio
nest_asyncio.apply()

import asyncio
import shutil
import os
import glob
import logging
from io import BytesIO
from uuid import uuid4

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import torch
from ultralytics import YOLO
from prompt import COMBINED_PROMPT, COMBINED_PROMPT2
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from dotenv import load_dotenv
import json

load_dotenv()
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Gemini imports
from google import genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def log_message(msg):
    st.sidebar.write(msg)

# -------------
# Session State
# -------------
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.final_json_results = []  # store final JSON for each page
    st.session_state.previous_pdf_uploaded = None

# -------------
# Directory Setup
# -------------
DATABASE_DIR = "Database"
if not os.path.exists(DATABASE_DIR):
    os.makedirs(DATABASE_DIR)

COMMERICIAL_DIR = os.path.join(DATABASE_DIR, "Commercial")
if not os.path.exists(COMMERICIAL_DIR):
    os.makedirs(COMMERICIAL_DIR)
    
RESIDENTIAL_DIR = os.path.join(DATABASE_DIR, "Residential")
if not os.path.exists(RESIDENTIAL_DIR):
    os.makedirs(RESIDENTIAL_DIR)
    

DATA_DIR = "data"
LOW_RES_DIR = os.path.join(DATA_DIR, "40_dpi")
HIGH_RES_DIR = os.path.join(DATA_DIR, "500_dpi")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)

################################################################################
# 1) PDF -> Low-Res & High-Res
################################################################################
def pdf_to_images(pdf_path, output_dir, fixed_length=1080):
    log_message(f"Converting PDF to images at fixed length {fixed_length}px...")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_message(f"Error opening PDF: {e}")
        raise

    file_paths = []
    for i in range(len(doc)):
        page = doc[i]
        scale = fixed_length / page.rect.width
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)
        image_filename = f"{base_name}_page_{i+1}.jpg"
        image_path = os.path.join(output_dir, image_filename)
        pix.save(image_path)
        log_message(f"Saved image: {image_path}")
        file_paths.append(image_path)
    doc.close()
    log_message("PDF conversion completed.")
    return file_paths

################################################################################
# 2) YOLO for Block Detection
################################################################################
class BlockDetectionModel:
    def __init__(self, weight, device=None):
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(weight).to(self.device)
        log_message(f"YOLO model loaded on {self.device}.")

    def predict_batch(self, images_dir):
        if not os.path.exists(images_dir) or not os.listdir(images_dir):
            raise ValueError(f"Directory {images_dir} is empty or does not exist.")
        images = glob.glob(os.path.join(images_dir, "*.jpg"))
        log_message(f"Found {len(images)} images for detection.")

        output = {}
        batch_size = 10
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            log_message(f"YOLO detection on images {i+1}-{min(i+batch_size, len(images))} of {len(images)}.")
            results = self.model(batch)
            for result in results:
                image_name = os.path.basename(result.path)
                labels = result.boxes.cls.tolist()
                boxes = result.boxes.xywh.tolist()
                output[image_name] = [
                    {"label": label, "bbox": box}
                    for label, box in zip(labels, boxes)
                ]
        return output

def scale_bboxes(bbox, src_size=(662, 468), dst_size=(4000,3000)):
    scale_x = dst_size[0] / src_size[0]
    scale_y = scale_x
    return bbox[0]*scale_x, bbox[1]*scale_y, bbox[2]*scale_x, bbox[3]*scale_y

def crop_and_save(detection_output, output_dir):
    log_message("Cropping detected regions from high-res images...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    final_data = {}
    for image_name, detections in detection_output.items():
        image_resource_path = os.path.join(output_dir, image_name.replace(".jpg",""))
        image_path = os.path.join(HIGH_RES_DIR, image_name)
        if not os.path.exists(image_path):
            log_message(f"High-res image missing: {image_path}")
            continue
        if not os.path.exists(image_resource_path):
            os.makedirs(image_resource_path)

        try:
            with Image.open(image_path) as image:
                cropped_info = {}
                for det in detections:
                    label = det["label"]
                    bbox = det["bbox"]
                    label_dir = os.path.join(image_resource_path, str(label))
                    os.makedirs(label_dir, exist_ok=True)
                    x, y, w, h = scale_bboxes(bbox)
                    cropped_img = image.crop((x - w/2, y - h/2, x + w/2, y + h/2))
                    cropped_name = f"{label}_{len(os.listdir(label_dir))+1}.jpg"
                    cropped_path = os.path.join(label_dir, cropped_name)
                    cropped_img.save(cropped_path)
                    cropped_info.setdefault(label, []).append(cropped_path)
                cropped_info["Image_Path"] = image_path  # entire page
                final_data[image_name] = cropped_info
                log_message(f"Cropped images saved for {image_name}")
        except Exception as e:
            log_message(f"Error cropping {image_name}: {e}")

    return final_data

################################################################################
# 3) Single Gemini Call with Entire Page + Cropped
################################################################################
def gemini_call_entire_and_crops(entire_path, cropped_paths, single_prompt):
    """
    Send a single prompt plus the entire page image (first), 
    then each cropped region image. Let Gemini handle them in one call.
    """
    contents = [single_prompt]
    # 1) Entire page
    try:
        with Image.open(entire_path) as img:
            page_resized = img.resize((int(img.width/2), int(img.height/2)))
            contents.append(page_resized)
    except Exception as e:
        print(f"Error resizing entire page: {e}")

    # 2) All cropped blocks
    for cpath in cropped_paths:
        try:
            with Image.open(cpath) as img:
                cropped_resized = img.resize((int(img.width/2), int(img.height/2)))
                contents.append(cropped_resized)
        except Exception as e:
            print(f"Error resizing cropped region {cpath}: {e}")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents
    )
    resp_text = response.text.strip()
    if resp_text.startswith("```"):
        resp_text = resp_text.replace("```","").strip()
        if resp_text.lower().startswith("json"):
            resp_text = resp_text[4:].strip()
    log_message(f"Gemini response for {entire_path}")

    # Attempt JSON
    try:
        return json.loads(resp_text)
    except json.JSONDecodeError:
        return resp_text  # fallback raw  


# Function to classify the images and create folder structure
def create_folders_and_classify(json_data, image_path):
    """
    Classify the image based on JSON data and create the folder structure:
    Root Folder -> Building_Purpose -> Project_Title -> Drawing_Type -> Image
    """
    # Extract classification data
    building_purpose = json_data.get("Purpose_of_Building", "Other").capitalize()
    project_title = json_data.get("Project_Title", "Unknown_Project")
    drawing_type = json_data.get("Drawing_Type", "unknown").lower()

    # Define the folder structure
    building_purpose_dir = os.path.join(DATABASE_DIR, building_purpose)
    project_title_dir = os.path.join(building_purpose_dir, project_title)
    drawing_type_dir = os.path.join(project_title_dir, drawing_type)

    # Ensure all directories are created
    os.makedirs(drawing_type_dir, exist_ok=True)

    # Define new image path based on the folder structure
    image_filename = os.path.basename(image_path)
    new_image_path = os.path.join(drawing_type_dir, image_filename)

    # Move the image to the classified folder
    try:
        shutil.copy(image_path, new_image_path)  # Use copy instead of move to avoid accidental loss
        log_message(f"Image {image_filename} moved to {drawing_type_dir}.")
    except Exception as e:
        log_message(f"Error moving image {image_filename}: {e}")

    return new_image_path  # Return the new path for the image


################################################################################
# The Pipeline & Streamlit UI
################################################################################
st.sidebar.title("PDF Processing")
st.title("PDF Analyzer v1.0")
st.write("Powered by Streamlit, YOLO, and Gemini AI.")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    if uploaded_pdf.name != st.session_state.previous_pdf_uploaded:
        st.session_state.processed = False
        st.session_state.final_json_results = []
        st.session_state.previous_pdf_uploaded = uploaded_pdf.name

    pdf_path = os.path.join(DATA_DIR, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success("PDF uploaded successfully.")

if uploaded_pdf and not st.session_state.processed:
    if st.sidebar.button("Run Processing Pipeline"):
        # st.info("Converting PDF to low-res & high-res images...")
        low_res_list = pdf_to_images(pdf_path, LOW_RES_DIR, fixed_length=662)
        high_res_list = pdf_to_images(pdf_path, HIGH_RES_DIR, fixed_length=4000)

        # st.info("Running YOLO detection on low-res images...")
        model = BlockDetectionModel("best_small_yolo11_block_etraction.pt")
        detection_results = model.predict_batch(LOW_RES_DIR)

        # st.info("Cropping from high-res images...")
        page_crops = crop_and_save(detection_results, OUTPUT_DIR)

        # Single prompt for entire + cropped images
        # The prompt can clarify: "The 1st image is entire page, the next images are cropped regions, etc."
        gemini_prompt = COMBINED_PROMPT2
        final_results = []

        # For each page, gather entire page + all cropped
        for page_key, block_info in page_crops.items():
            entire_image = block_info["Image_Path"]
            # Collect all cropped images
            all_crops = []
            for label_id, path_list in block_info.items():
                if label_id == "Image_Path":
                    continue
                all_crops.extend(path_list)

            # single gemini call
            single_json = gemini_call_entire_and_crops(entire_image, all_crops, gemini_prompt)
            # Store result
            final_results.append({
                "page_name": entire_image,
                "gemini_output": single_json
            })

        st.session_state.final_json_results = final_results
        st.session_state.processed = True
        st.success("Processing complete! See results below.")

if uploaded_pdf and st.session_state.processed:
    st.header("Final JSON Results per Page")
    for idx, item in enumerate(st.session_state.final_json_results):
        # Extract image path and JSON data
        image_path = item['page_name']
        json_data = item['gemini_output']
        
        # Create folders and classify the image
        classified_image_path = create_folders_and_classify(json_data, image_path)
        
        # Display the classified image and JSON
        st.image(classified_image_path, caption=f"Page {idx+1} - {json_data['Purpose_of_Building']}", use_container_width =True)
        st.json(json_data)  # Show the full JSON for the page
        

