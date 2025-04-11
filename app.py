import streamlit as st
import fitz  # PyMuPDF
import asyncio
import nest_asyncio
nest_asyncio.apply()

import os
import json
import tempfile
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Gemini imports (adjust for your environment)
from google import genai

# LangChain & FAISS imports
import faiss
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# BM25 + Rerank
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from nltk.tokenize import word_tokenize

# Load environment variables (Gemini & OpenAI keys)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

###############################################################################
# 1. PDF -> High-DPI Images (in memory)
###############################################################################
def pdf_to_images(pdf_bytes, dpi=400):
    """
    Converts each page of the PDF (in memory) to a PIL Image at the specified DPI.
    Returns a list of PIL Images.
    """
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=dpi)
            # Convert to PIL
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

###############################################################################
# 2. Async call to Gemini
###############################################################################
async def analyze_image_gemini(img: Image, prompt: str):
    """
    Sends (img + prompt) to Gemini, returns the raw text or JSON as string.
    """
    # Convert PIL image to bytes
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    # Wrap synchronous Gemini call in asyncio.to_thread
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.0-flash",
        contents=[img, prompt]  # or [buffer, prompt] if that works in your environment
    )
    return response.text

###############################################################################
# 3. Convert Gemini responses -> Documents
###############################################################################
def build_documents_from_responses(responses):
    docs = []
    for idx, resp_text in enumerate(responses):
        # Strip potential code fences
        clean_text = resp_text.strip().strip("```").strip()
        try:
            data = json.loads(clean_text)
            content_str = json.dumps(data)
        except json.JSONDecodeError:
            content_str = resp_text

        doc = Document(
            page_content=content_str,
            metadata={"page_number": idx + 1}
        )
        docs.append(doc)
    return docs

###############################################################################
# 4. Build a Vector Store in memory
###############################################################################
def build_vector_store(docs):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )
    example_vec = embeddings.embed_query("test")
    dimension = len(example_vec)

    faiss_index = faiss.IndexFlatL2(dimension)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vectorstore.add_documents(docs)
    return vectorstore

###############################################################################
# 5. Create an Ensemble Retriever (BM25 + Vector + Re-rank)
###############################################################################
def create_ensemble_retriever(docs, vectorstore):
    bm25 = BM25Retriever.from_documents(docs, k=10, preprocess_func=word_tokenize)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    ensemble = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.6, 0.4])
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble
    )
    return final_retriever

###############################################################################
# 6. The Streamlit App
###############################################################################
def main():
    st.title("PDF + Gemini + Vector Search (Streamlit Cloud)")

    # Hard-coded Gemini system prompt
    system_prompt  = """
        You are an advanced AI system specialized in analyzing architectural and engineering drawings in detail.

        Please return a single, valid JSON object that begins with:
        {
        "Drawing_Type": "<floor_plan | section | elevation | detail | other>"
        ...
        }

        Based on the recognized drawing type, fill in relevant details. If you only see partial labels or geometry, do your best to infer. If data is missing or cannot be deduced, use "N/A" or 0.

        ==================================================================================
        IF THE DRAWING IS A FLOOR PLAN:
        ==================================================================================
        1. "Building_Use": 
        - "Residential", "Commercial", "Mixed-use", or "N/A" if unclear.
        - If partial signage suggests a certain building type, use that.

        2. "Space_Classification": 
        - A list of areas categorized as:
            - Communal (hallways, lounges, staircases, elevator lobbies)
            - Private (bedrooms, apartments, bathrooms)
            - Service (kitchens, utility rooms, storage)
        - If unsure, mark "N/A". If text references or shapes suggest certain areas, name them.

        3. "Number_of_Units": 
        - Total identifiable apartments/units. 
        - If unlabeled but repeated shapes appear, estimate.

        4. "Number_of_Stairs": 
        - Count any staircases (look for text like “Stair”, “S”, or typical stair icons).
        - If you suspect partial stair references, try to confirm visually. If none, return 0.

        5. "Number_of_Elevators": 
        - Count any spaces that appear to be elevator shafts (icons or partial text). 
        - If found but not labeled, guess if it looks like an elevator.

        6. "Number_of_Hallways": 
        - Corridors connecting multiple areas. If unlabeled but shape indicates a corridor, include it.

        7. "Unit_Details": 
        - A list of objects, one for each distinct unit/apartment.
        - For each unit:
            {
            "Unit_Number": "If text says A-1, B-2, APT-2B, etc., use that; else 'N/A'",
            "Unit_Area": "Try to approximate if dimension lines or a scale bar is visible, else 'N/A'",
            "Bedrooms": "Attempt to infer from text or repeated room labels. If unknown, 0 or guess (1,2).",
            "Bathrooms": "Similarly, attempt to identify from partial labeling or geometry. If none, 0.",
            "Has_Living_Room": true/false if you see references or shape typical of living spaces,
            "Has_Kitchen": true/false if you see references or shape typical of a kitchen,
            "Has_Balcony": true/false if balcony text or shape is visible,
            "Special_Features": ["study room", "utility room", etc., if recognized; else empty list]
            }

        8. "Stairs_Details": 
        - A list for each staircase:
            {
            "Location": "e.g., near unit 1, center, corner, etc.",
            "Purpose": "typical usage, e.g., access to upper floors or emergency exit"
            }

        9. "Elevator_Details": 
        - A list describing each elevator:
            {
            "Location": "e.g., near unit 3, center of the plan",
            "Purpose": "vertical transportation"
            }

        10. "Hallways": 
            - A list of corridor objects:
            {
                "Location": "brief text (connects units 2 and 3, or center hallway)",
                "Approx_Area": "if dimension lines exist, approximate area; else 'N/A'"
            }

        11. "Other_Common_Areas": 
            - A list describing any additional communal spaces:
            {
                "Area_Name": "entrance hall, lobby, courtyard, lounge, etc.",
                "Approx_Area": "estimate if dimension lines appear, else 'N/A'"
            }

        ==================================================================================
        IF THE DRAWING IS A SECTION VIEW:
        ==================================================================================
        1. "Vertical_Information": 
        - A list capturing visible floors, basements, or mezzanines:
            {
            "Floor_Label": "e.g., Basement, Ground, 1st Floor, or 'N/A' if unclear",
            "Floor_Height": "approx or 'N/A' if unlabeled",
            "Ceiling_Height": "approx or 'N/A' if unlabeled"
            }

        2. "Material_Details": 
        - Summarize materials or finishes if text references (e.g., 'concrete', 'steel'), or guess from standard notation.

        3. "Structural_Elements": 
        - A list of recognized columns, beams, load-bearing walls, slabs, etc.

        4. "Room_Layout": 
        - Insights into internal partitions as shown in section (rooms, corridors, big open spaces).

        5. "Other_Key_Features": 
        - Windows, doors, ventilation shafts, plumbing lines, etc.

        ==================================================================================
        IF THE DRAWING IS AN ELEVATION OR A DETAIL:
        ==================================================================================
        - ELEVATION:
        - "Facade_Elements": e.g., windows, doors, balconies, decorative features
        - "Approx_Building_Height": e.g., "30 m" if dimension lines exist, else 'N/A'
        - "Comments_or_Notes": Additional orientation or material info

        - DETAIL:
        - "Detail_Description": e.g., window frame detail, rebar joint, façade cross-section
        - "Material_Notes": thickness, insulation, adhesives, finishing layers
        - "Comments_or_Notes": Additional clarifications (fasteners, anchors, special notation)

        ==================================================================================
        GENERAL GUIDELINES:
        ==================================================================================
        - If the drawing does not match any known category, "Drawing_Type": "other".
        - If data is missing or cannot be inferred, use "N/A" or 0.
        - Return ONLY the JSON object, no code fences or commentary.
        - The first key is "Drawing_Type" with one of: floor_plan, section, elevation, detail, other.
        - Provide as much detail as possible, even from partial dimension lines or partial text references.
        - Attempt to differentiate units if you suspect multiple types with different bedroom or bathroom counts.
        """

    # Step 1: PDF Upload
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if not uploaded_pdf:
        st.info("Please upload a PDF to begin.")
        return

    # Step 2: Button to process the PDF
    if st.button("Process PDF"):
        with st.spinner("Reading PDF and converting to images..."):
            pdf_bytes = uploaded_pdf.read()

            # Convert PDF -> images in memory
            images = pdf_to_images(pdf_bytes, dpi=400)

        st.write(f"PDF has {len(images)} pages.")

        # Asynchronously call Gemini on each page
        async def process_pages():
            tasks = []
            for page_img in images:
                tasks.append(analyze_image_gemini(page_img, system_prompt))
            results = await asyncio.gather(*tasks)
            return results

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            responses = loop.run_until_complete(process_pages())
        finally:
            loop.close()

        # Turn responses into Documents
        docs = build_documents_from_responses(responses)
        if not docs:
            st.warning("No text/JSON returned from Gemini.")
            return

        # Build in-memory vector store
        vectorstore = build_vector_store(docs)

        # Build the retriever
        retriever = create_ensemble_retriever(docs, vectorstore)

        st.session_state["retriever"] = retriever
        st.session_state["docs"] = docs
        st.success("Processing complete! You can now query the results below.")

    # Step 3: Query the results
    if "retriever" in st.session_state and st.session_state["retriever"]:
        query = st.text_input("Enter your query to search in the Gemini-extracted JSON:")
        if query:
            st.write("Searching...")
            results = st.session_state["retriever"].invoke(query)

            if results:
                st.write(f"Found {len(results)} matching pages.")
                for i, doc in enumerate(results):
                    st.markdown(f"### Result {i+1} (Page {doc.metadata.get('page_number')}):")
                    try:
                        parsed = json.loads(doc.page_content)
                        st.json(parsed)
                    except:
                        st.code(doc.page_content)
            else:
                st.write("No relevant results found.")

if __name__ == "__main__":
    main()
