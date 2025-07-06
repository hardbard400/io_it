# pdf_processor.py

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

def process_pdf_to_json(pdf_file_path: str):
    """
    Uploads a PDF file to the Gemini API, processes its content into a 
    structured JSON format, and returns the data as a Python list of records.

    Args:
        pdf_file_path: The local path to the PDF file.

    Returns:
        A list of dictionaries, where each dictionary is a record, 
        or None if an error occurs.
    """
    load_dotenv()
    gemini_api = os.getenv("GEMINI_API_KEY")
    if not gemini_api:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return None
    
    genai.configure(api_key=gemini_api)

    print(f"Uploading and processing file: {pdf_file_path}...")
    try:
        pdf_file = genai.upload_file(path=pdf_file_path)
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    prompt = """
    Your task is to process the uploaded PDF document and convert its content into a structured JSON format for embedding.

    Follow these instructions:
    1.  Thoroughly analyze the entire document.
    2.  Break down the content into concise, self-contained statements or sentences. Each statement should represent a distinct piece of information.
    3.  Generate a JSON array of records. Each record must have the following fields:
        * `chunk_text`: The statement you generated from the PDF's content.
        * `category`: Use "document_analysis" for this field for all records.
    4.  The final output must be a single JSON object containing a list named "records". Do not include the `_id` field.

    Example format:
    {
      "records": [
        {
          "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.",
          "category": "document_analysis"
        },
        {
          "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.",
          "category": "document_analysis"
        }
      ]
    }
    """

    print("Generating structured content from PDF...")
    try:
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content([pdf_file, prompt], generation_config=generation_config)
        
        # Clean up the uploaded file on the server
        genai.delete_file(pdf_file.name)
        print(f"Cleaned up uploaded file: {pdf_file.name}")

        records_data = json.loads(response.text)
        return records_data.get("records", [])

    except Exception as e:
        print(f"An error occurred during content generation or processing: {e}")
        # Attempt to clean up the file even if generation fails
        try:
            genai.delete_file(pdf_file.name)
            print(f"Cleaned up uploaded file after error: {pdf_file.name}")
        except Exception as cleanup_error:
            print(f"Error during file cleanup after an exception: {cleanup_error}")
            
        return None