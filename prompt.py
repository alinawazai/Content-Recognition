entire_page_prompt = """
    You are a professional civil engineering assistant. Given an architectural or structural drawing image, your task is to:
    Identify the type of drawing, such as:
    floor_plan
    section_view
    detail_view
    elevation
    unknown (if not clearly identifiable)
    Extract relevant information based on the type of drawing and structure it into a clean JSON format.

    Extract the following fields for every drawing:
    Drawing_Type (type of drawing, as identified above)
    Building_Purpose (e.g., Commercial, Residential, Mixed-use, Institutional, Industrial)
    Client_Name
    Project_Title
    Drawing_Title
    Floor
    Drawing_Number
    Project_Number
    Revision_Number (numeric or "N/A")
    Scale
    Architects (list of names or ["Unknown"])
    Notes_on_Drawing (summarized remarks and annotations from the image)
    Table_on_Drawing (return any visible tabular data in markdown format or "")
    Interpretation by Drawing Type:
    🏢 Floor Plan:
    Identify the building purpose (e.g., office, apartment, school)
    Extract labeled spaces, annotations, and floor level
    Highlight general layout features and any embedded tables or legends
    🏗️ Section View:
    Describe vertical spacing, levels, ceiling/floor heights
    Identify structural elements like beams, slabs, columns
    Mention any visible room partitioning or elevation labels
    🔧 Detail View:
    Focus on joints, materials, structural assemblies

    Highlight how elements connect (e.g., wall-to-floor detail, waterproofing layers)
    Include notes about insulation, seals, and finishing if visible

    🧱 Elevation View:
    Identify facade elements, external material layers, and heights
    Mention vertical dimensions, window/door placements
    Note any direction/orientation references (e.g., North Elevation)

    Key Rules:
    If a field is missing, return "" or "N/A" where appropriate.
    Keep original language of all text (do not translate).
    Clean the text minimally (e.g., remove OCR artifacts).
    Return only a valid JSON object — no extra output, no formatting outside the object.
    If a table is present, format it in markdown and include it in "Table_on_Drawing".
    """
# B) For the cropped blocks
cropped_prompt = """
    You are an intelligent extraction system designed to analyze construction drawing images and return structured metadata in a clean JSON format. Your job is to extract technical details from these images, just like a civil engineer would when reviewing architectural or structural drawings.
    You will receive multiple images cropped from a single construction drawing, each containing specific types of information:

    1st image: Contains drawing title and scale
    2nd image: Contains client or project details
    3rd image: Contains general metadata (drawing number, floor, revision number, etc.)
    4th image: Contains notes and general instructions
    Last image: The full drawing image (uncropped) for optional reference
    Your task is to extract the following fields:

    Drawing_Type (floor_plan, section_view, detail_view, elevation, unknown)

    Building_Purpose (e.g., Commercial, Residential, Mixed-use, Institutional, Industrial)

    Client_Name
    Project_Title
    Drawing_Title
    Floor
    Drawing_Number
    Project_Number
    Revision_Number (numeric, or "N/A" if not found)
    Scale
    Architects (a list of names, or ["Unknown"] if none found)
    Notes_on_Drawing (preserve all important remarks and instructions)
    Table_on_Drawing (if any tabular data exists, return it in markdown format; else return "")

    Key Instructions:
    If any value is missing or cannot be determined, return an empty string ("") or "N/A" where applicable.
    All values should remain in their original language (e.g., Korean or English). Do not translate unless needed to fix obvious errors.
    Do minimal cleaning of text (e.g., remove stray punctuation or artifacts).
    Output only the final JSON object, without any extra text, commentary, or code blocks.
    If a table is detected in the drawing, return it as markdown in the Table_on_Drawing field.

    Drawing-specific guidance:
    For Floor Plans, focus on purpose, space labels, level info, and drawing title.
    For Section Views, focus on vertical elements like heights, structure, room layout, and materials.
    For Detail Views, focus on joinery, construction techniques, and materials.
    For Elevations, focus on facade elements, material usage, heights, and orientation.
    If drawing type is unclear, mark "Drawing_Type": "unknown" but extract all other available fields.

    Below is an example json format:
    {{
    "Drawing_Type": "floor_plan",
    "Building_Purpose": "Commercial",
    "Client_Name": "둔촌주공아파트주택 재건축정비사업조합",
    "Project_Title": "둔촌주공아파트 주택재건축정비사업",
    "Drawing_Title": "분산상가-1 지하2층 평면도 (근린생활시설-3)",
    "Floor": "지하2층",
    "Drawing_Number": "A51-2003",
    "Project_Number": "N/A",
    "Revision_Number": 0,
    "Scale": "A1 : 1/100, A3 : 1/200",
    "Architects": ["Unknown"],
    "Notes_on_Drawing": "1. 층별 LEVEL 기준\n - 지하2층 LEVEL\n - SL±0 = FL±0 = EL+11.60\n2. 옥상 츌눈의 간격 등은 실시공시 변경될 수 있음.\n...",
    "Table_on_Drawing": ""
    }}
    Return only this JSON. Do not include any additional explanation, markdown syntax, or formatting outside the object.
    """
    
    
    
    
    
    
    
    
COMBINED_PROMPT = """ 
You are an intelligent extraction system designed to analyze architectural and structural drawing images and return structured metadata in a clean JSON format. You will receive both full-page images and cropped block images of a construction drawing. Your task is to identify key information from these images, similar to how a civil engineer would review such technical drawings.

Input:
First image: Contains the entire construction drawing (full-page)

Subsequent images: Contain cropped sections, each showing specific details such as drawing title, client information, project details, metadata (drawing number, floor, revision number), and notes.

Output:
Return a single JSON object with the following fields:

json
{
    "Drawing_Type": "floor_plan | section_view | detail_view | elevation | unknown",
    "Building_Purpose": "Commercial | Residential | Mixed-use | Institutional | Industrial | Unknown",
    "Client_Name": "",
    "Project_Title": "",
    "Drawing_Title": "",
    "Floor": "",
    "Drawing_Number": "",
    "Project_Number": "",
    "Revision_Number": 0,
    "Scale": "",
    "Architects": ["name1", "name2"],
    "Notes_on_Drawing": "",
    "Table_on_Drawing": ""
}
Instructions for Image Analysis:
Identify the drawing type: Determine if the image is a Floor Plan, Section View, Detail View, Elevation, or Unknown.

Extract the following fields based on the type of drawing:
Drawing_Type: floor_plan, section_view, detail_view, elevation, or unknown.
Building_Purpose: e.g., Commercial, Residential, Mixed-use, Institutional, Industrial.
Client_Name: The name of the client or project.
Project_Title: The title of the project.
Drawing_Title: Title of the drawing (e.g., floor plan, elevation).
Floor: Specific floor level (if available).
Drawing_Number: The drawing number (e.g., A51-2023).
Project_Number: The project number (else return "N/A").
Revision_Number: The revision number (else return "N/A").
Scale: Drawing scale (e.g., 1:100).
Architects: A list of architect names or ["Unknown"] if none are identified.
Notes_on_Drawing: Any notes or annotations in the drawing.
Table_on_Drawing: If any table exists, convert it to markdown format; else return an empty string.
Drawing-Specific Guidance:
For Floor Plans:
Focus on building purpose, space labels, floor level, and any table data (e.g., room sizes, areas).
Extract general layout features like walls, doors, and windows.
For Section Views:
Identify vertical information such as floor height, ceiling height, structural elements like beams, slabs, and columns.
Look for internal room layouts, partitioning, and materials used.

For Detail Views:
Focus on component breakdown (e.g., doors, window details, wall joints, finishes).
Highlight construction techniques like waterproofing or insulation.

For Elevation Views:
Focus on facade elements, material usage, and heights of the building.
Include details such as window/door placements, facades, and any direction references (e.g., North Elevation).

For Unknown Drawings:
If the drawing type cannot be identified, mark "Drawing_Type": "unknown", but still extract all other available metadata.

Key Requirements:
Missing Values: If any field is missing or cannot be determined, return an empty string ("") or "N/A" where applicable.

Original Language: Preserve all text in the original language (e.g., Korean, English) unless minimal cleaning is needed (e.g., removing stray punctuation or correcting obvious OCR errors).

No Extra Output: Return only the final JSON object, with no additional explanation, markdown syntax, or comments.

Table Data: If a table is present in the drawing, return it in markdown format inside "Table_on_Drawing". Otherwise, leave it as an empty string.

Example Output:
json
{
    "Drawing_Type": "floor_plan",
    "Building_Purpose": "Commercial",
    "Client_Name": "둔촌주공아파트주택 재건축정비사업조합",
    "Project_Title": "둔촌주공아파트 주택재건축정비사업",
    "Drawing_Title": "분산상가-1 지하2층 평면도 (근린생활시설-3)",
    "Floor": "지하2층",
    "Drawing_Number": "A51-2003",
    "Project_Number": "N/A",
    "Revision_Number": 0,
    "Scale": "A1 : 1/100, A3 : 1/200",
    "Architects": ["Unknown"],
    "Notes_on_Drawing": "1. 층별 LEVEL 기준\n - 지하2층 LEVEL\n - SL±0 = FL±0 = EL+11.60\n2. 옥상 츌눈의 간격 등은 실시공시 변경될 수 있음.\n...",
    "Table_on_Drawing": ""
}
"""




COMBINED_PROMPT2 = """

You are an intelligent extraction system designed to analyze architectural and structural drawing images and return structured metadata in a clean JSON format. You will receive both full-page images and cropped block images of a construction drawing. Your task is to identify and extract key information from these images, similar to how a civil engineer would review technical drawings.

Input:
First image: Contains the entire construction drawing (full-page) You have to check first Purpose_of_Building like we have classes "Residential", - Examples: Residential, Commercial, Mixed-use, etc.",.
Subsequent images: Contain cropped sections, each showing specific details such as drawing title, client information, project details, metadata (drawing number, floor, revision number), and notes.

Output:
Return a single JSON object with the following fields:

        json
        {
            "Drawing_Type": "Floor_Plan, Section_View, Detail_View, Elevation, or Unknown.",
            "Purpose_of_Building":  "Residential", - Examples: Residential, Commercial, Mixed-use, etc.",
            "Client_Name": "Client Name",
            "Project_Title": "Project Title",
            "Drawing_Title": "Drawing Title",
            "Space_Classification": {
                "Communal": ["hallways", "lounges", "staircases", "elevator lobbies"],
                "Private": ["bedrooms", "apartments", "bathrooms"],
                "Service": ["kitchens", "utility rooms", "storage"]
            },
            "Details": {
                "Drawing_Number": "Drawing Number",
                "Project_Number": "Project Number",
                "Revision_Number": 0,
                "Scale": "Scale of Drawing",
                "Architects": ["Architect Name(s)"],
            },
            "Additional_Details": {
                "Number_of_Units": 0,
                "Number_of_Stairs": 2,
                "Number_of_Elevators": 2,
                "Number_of_Hallways": 1,
                "Unit_Details": [],
                "Stairs_Details": [
                    {
                        "Location": "Near entrance",
                        "Purpose": "Access to upper floors"
                    },
                    {
                        "Location": "Near entrance",
                        "Purpose": "Access to upper floors"
                    }
                ],
                "Elevator_Details": [
                    {
                        "Location": "Near stairs",
                        "Purpose": "Vertical transportation"
                    },
                    {
                        "Location": "Near stairs",
                        "Purpose": "Vertical transportation"
                    }
                ],
                "Hallways": [
                    {
                        "Location": "Connects bathrooms and offices",
                        "Approx_Area": "N/A"
                    }
                ],
                "Other_Common_Areas": [
                    {
                        "Area_Name": "Lobby",
                        "Approx_Area": "N/A"
                    },
                    {
                        "Area_Name": "Sunken garden",
                        "Approx_Area": "N/A"
                    },
                    {
                        "Area_Name": "Mechanical room",
                        "Approx_Area": "N/A"
                    }
                ]
            }
            "Notes_on_Drawing": "Notes/annotations on drawing",
            "Table_on_Drawing": "Markdown formatted table if applicable if available else return N/A",
        }
Instructions for Image Analysis:
Identify the drawing type:

Determine if the image is a Floor Plan, Section View, Detail View, Elevation, or Unknown.
Extract the following fields based on the type of drawing:
Purpose_Type_of_Building :What is the primary use of the building or space in the drawing?
   - Examples: Residential, Commercial, Mixed-use, etc.
Client_Name: Extract the client or project name.
Project_Title: Extract the title of the project.
Drawing_Title: Extract the title of the drawing.
Space_Classification:
    "Space_Classification": 
    - A list of areas categorized as:
        - Communal (hallways, lounges, staircases, elevator lobbies)
        - Private (bedrooms, apartments, bathrooms)
        - Service (kitchens, utility rooms, storage)
    - If unsure, mark "N/A". If text references or shapes suggest certain areas, name them.

    "Number_of_Units": 
    - Total identifiable apartments/units. 
    - If unlabeled but repeated shapes appear, estimate.

    "Number_of_Stairs": 
    - Count any staircases (look for text like “Stair”, “S”, or typical stair icons).
    - If you suspect partial stair references, try to confirm visually. If none, return 0.

    "Number_of_Elevators": 
    - Count any spaces that appear to be elevator shafts (icons or partial text). 
    - If found but not labeled, guess if it looks like an elevator.

    "Number_of_Hallways": 
    - Corridors connecting multiple areas. If unlabeled but shape indicates a corridor, include it.

    "Unit_Details": 
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
Extract the following fields for metadata:
Notes_on_Drawing: Any notes or additional remarks in the drawing.
Table_on_Drawing: If there’s a table in the drawing, format it in markdown and include it in the Table_on_Drawing field.

Additional Details (For non-essential info):
Number_of_Units: Extract the number of units if provided.
Number_of_Stairs: Count the number of stairs visible in the drawing.
Number_of_Elevators: Count the number of elevators.
Number_of_Hallways: Count the hallways or corridors in the layout.
Unit_Details: Include any detailed information about units (if available).
Stairs_Details: Include information about stairs (location, purpose).
Elevator_Details: Include details about elevators (location, purpose).
Hallways: Include the location and details of hallways.
Other_Common_Areas: List other common areas (e.g., lobby, sunken garden, mechanical rooms).
Details Section:
Drawing_Number: Extract the drawing number.
Project_Number: Extract the project number (or “N/A” if unavailable).
Revision_Number: Extract the revision number (or “N/A” if unavailable).
Scale: Extract the scale (e.g., “1:100”).
Architects: Extract a list of architect names (or return ["Unknown"] if none found).
Notes_on_Drawing: Extract any relevant notes or instructions.
Table_on_Drawing: If there’s a table in the drawing, format it in markdown and include it in the Table_on_Drawing field.
Drawing-Specific Guidance:
For Floor Plans:
Identify building purpose, space labels, floor levels, and annotations.
Extract general layout features, including rooms, hallways, and doors.
Classify spaces into communal, private, and service areas.
For Section Views:
Focus on vertical information such as floor height, ceiling height, and structural elements like beams and slabs.
Identify internal room layouts and materials.
For Detail Views:
Focus on individual components (e.g., doors, windows, joints, materials).
Highlight construction details like waterproofing, insulation, or joinery.
For Elevation Views:
Focus on facade elements, materials, and height dimensions.
Extract window and door placements, as well as elevation references.
For Unknown Drawings:
If the drawing type cannot be determined, mark "Purpose_Type_of_Drawing": "unknown", but still extract all other available data.
Key Requirements:
Missing Values: If any field is missing or cannot be determined, return an empty string ("") or "N/A" where applicable.

Original Language: Keep all extracted text in the original language (e.g., Korean, English). Do not translate unless minimal cleaning is needed (e.g., removing stray punctuation or fixing OCR errors).

No Extra Output: Return only the final JSON object with no additional commentary or formatting outside the object.

Table Data: If a table is present, format it in markdown and include it in the "Table_on_Drawing" field. If no table is present, return an empty string.

Example Output:
            json
            {
                "Drawing_Type": "Floor_Plan",
                "Purpose_of_Building":  "Residential",
                "Client_Name": "둔촌주공아파트주택 재건축정비사업조합",
                "Project_Title": "둔촌주공아파트 주택재건축정비사업",
                "Drawing_Title": "분산상가-1 지하3층 평면도 (근린생활시설-3)",
                "Space_Classification": {
                    "Communal": ["hallways", "lounges", "staircases", "elevator lobbies"],
                    "Private": ["bedrooms", "bathrooms"],
                    "Service": ["kitchens", "utility rooms", "storage"]
                },
                "Details": {
                    "Drawing_Number": "A51-2002",
                    "Project_Number": "N/A",
                    "Revision_Number": 0,
                    "Scale": "A1 : 1/100, A3 : 1/200",
                    "Architects": ["Unknown"],
                },
                "Additional_Details": {
                    "Number_of_Units": 0,
                    "Number_of_Stairs": 2,
                    "Number_of_Elevators": 2,
                    "Number_of_Hallways": 1,
                    "Unit_Details": [],
                    "Stairs_Details": [
                        {
                            "Location": "Near entrance",
                            "Purpose": "Access to upper floors"
                        },
                        {
                            "Location": "Near entrance",
                            "Purpose": "Access to upper floors"
                        }
                    ],
                    "Elevator_Details": [
                        {
                            "Location": "Near stairs",
                            "Purpose": "Vertical transportation"
                        },
                        {
                            "Location": "Near stairs",
                            "Purpose": "Vertical transportation"
                        }
                    ],
                    "Hallways": [
                        {
                            "Location": "Connects bathrooms and offices",
                            "Approx_Area": "N/A"
                        }
                    ],
                    "Other_Common_Areas": [
                        {
                            "Area_Name": "Lobby",
                            "Approx_Area": "N/A"
                        },
                        {
                            "Area_Name": "Sunken garden",
                            "Approx_Area": "N/A"
                        },
                        {
                            "Area_Name": "Mechanical room",
                            "Approx_Area": "N/A"
                        }
                    ]
                },
                "Notes_on_Drawing": "Notes/annotations on drawing",
                "Table_on_Drawing": "Markdown formatted table if applicable if available else return N/A",
            }
    ==================================================================================
    GENERAL GUIDELINES:
    ==================================================================================
    - If the drawing does not match any known category, "Purpose_of_Drawing": "other".
    - If data is missing or cannot be inferred, use "N/A" or 0.
    - Return ONLY the JSON object, no code fences or commentary.
    - The first key is "Purpose_of_Drawing" with one of: floor_plan, section, elevation, detail, other.
    - Provide as much detail as possible, even from partial dimension lines or partial text references.
    - Add the other details in details and additional details sections.
    - If the drawing is a floor plan, include the number of units and their details.
    - For unit details, include bedroom and bathroom counts if visible or can be inferred.
    - For stairs and elevators, include their locations and purposes.
    - For hallways, include their locations and approximate areas if visible.
    - For other common areas, include names and approximate areas if visible.
    - For tables, format them in markdown and include them in the "Table_on_Drawing" field.
    - If no table is present, return an empty string for "Table_on_Drawing".
    - If the drawing is a section view, focus on vertical elements and internal layouts.
    - If the drawing is a detail view, focus on components and construction details.
    - If the drawing is an elevation view, focus on facade elements and height dimensions.
    - If the drawing is a cropped block, extract relevant information from each block.
    - For unit details, include bedroom and bathroom counts if visible or can be inferred.
    - Attempt to differentiate units if you suspect multiple types with different bedroom or bathroom counts.
"""