import google.generativeai as genai
import pdfplumber
import os
import json
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time

def extract_and_mark_page_text(pages):
    """
    Extracts text from a list of pdfplumber page objects and embeds page markers.
    The page number used is the actual page number from the PDF document.
    """
    if not pages:
        return ""
    
    full_text = ""
    for page in pages:
        # Use the actual page number from the PDF
        page_number = page.page_number
        page_text = page.extract_text()
        if page_text:
            # Add a clear marker at the beginning of each page's content
            full_text += f"\n\n--- PAGE {page_number} ---\n\n" + page_text
    return full_text

def get_structured_data_from_gemini(api_key, text_chunk):
    """
    Sends a large text chunk to the Gemini API to identify headers and extract 
    their corresponding text into a structured JSON format.
    """
    try:
        genai.configure(api_key=api_key)
        # Using a model with a larger context window is ideal for this task
        model = genai.GenerativeModel('gemini-2.5-flash')

        # --- REFINED SYSTEM PROMPT ---
        system_prompt = f"""
### **SYSTEM PROMPT**

You are an expert data extractor specializing in academic textbooks. Your goal is to meticulously parse the provided text, identify specific headers, and extract the content that follows them into a structured JSON format. The text you will receive contains markers like `--- PAGE X ---` to indicate where a new page begins.

---

### **Primary Task**

Analyze the text provided in the `{text_chunk}` and extract content based on the following rules.

---

### **Extraction and Content Rules**

**1. Identify Valid Headers:**
You must identify and extract content under the following header types:
* **Numbered Headers:** e.g., "1.1 Introduction", "2.5 The Cell Cycle".
* **Question-Based Headers:** e.g., "What is Biology?".
* **Descriptive Headers:** e.g., "Applications of Biotechnology", "Summary".

**2. Extract Associated Content:**
* For each valid header, extract ALL text that follows it until the next valid header begins.
* Content for a single topic may span multiple pages. Use the `--- PAGE X ---` markers to track this but continue extracting the text as one block.

**3. Page Number Rule:**
* For each record, the `"page_number"` MUST be the integer number from the `--- PAGE X ---` marker that immediately precedes or contains the topic's header.

**4. Strict Exclusion Rules (Content to IGNORE):**
* **DO NOT** extract: "UNIT 1", "Contents", "Table of Contents", "Figure 1.1", "Table 2.3", etc.
* **DO NOT** extract assessment or activity headers like: "Activity", "KEY WORDS", "Key Terms", "Review Questions", "End of unit questions".
* **DO NOT** extract learning objectives (e.g., "By the end of this section, you will be able to...").
* **DO NOT** include the `--- PAGE X ---` markers in your final output text.

---

### **Required Output Format**

Your response **MUST** be a single, valid JSON list of objects. If no valid content is found, return an empty list `[]`. **DO NOT** wrap the JSON in markdown (```json ... ```) or provide any explanatory text. Your entire response must be ONLY the JSON list.

**Example of a Valid JSON Output:**
```json
[
  {{
    "_id": "rec_1",
    "topic": "1.1 What is biotechnology?",
    "chunk_text": "Biotechnology is the use of micro-organisms to make things that people want, often involving industrial production. It is a field of applied biology that involves the use of living organisms and bioprocesses in engineering, technology, medicine and other fields requiring bioproducts.",
    "page_number": 4
  }},
  {{
    "_id": "rec_2",
    "topic": "The Scientific Method",
    "chunk_text": "The scientific method is a systematic process that involves observation, measurement, experiment, and the formulation, testing, and modification of hypotheses. This process is fundamental to scientific investigation and discovery.",
    "page_number": 5
  }}
]
```"""
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(system_prompt, safety_settings=safety_settings)
        
        # Basic validation of the response
        if not response.text:
            return None, "API returned an empty response."

        return response.text.strip(), None
    except Exception as e:
        return None, f"An error occurred with the Gemini API: {e}"

def save_data_to_json(data_list, output_path):
    """Saves the list of dictionaries to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4)
        return True, None
    except Exception as e:
        return False, f"Error saving to JSON file: {e}"
def main():
    """Main function to process PDF units and extract structured content."""
    
    load_dotenv()
    YOUR_API_KEY = os.getenv("gemma_gemini_api") 
    INPUT_PDF_PATH = "/workspaces/io_it/pdf's/Grade-9-Biology-Textbook.pdf"
    OUTPUT_JSON_PATH = "Grade_9_Biology_structured_content_3.json"
    
    # --- NEW: Define a character limit for a single API call ---
    # We use a safe limit to leave room for the prompt and the model's response.
    MAX_CHARS_PER_CALL = 90000

    PAGE_CHUNKS = {
        # "Unit 1": {"start": 3, "end": 13},
        # "Unit 2": {"start": 17, "end": 51},
        "Unit 3": {"start": 54, "end": 121},
        # "Unit 4": {"start": 127, "end": 169},
        # "Unit 5": {"start": 175, "end": 200},
        # "Unit 6": {"start": 204, "end": 228}
    }

    if not YOUR_API_KEY:
        print("Error: Gemini API key not found in .env file.")
        return

    if not os.path.exists(INPUT_PDF_PATH):
        print(f"Error: PDF file not found at '{INPUT_PDF_PATH}'")
        return

    all_structured_data = []

    print("--- Starting PDF Processing by Unit ---")
    try:
        with pdfplumber.open(INPUT_PDF_PATH) as pdf:
            for chunk_name, pages in PAGE_CHUNKS.items():
                start_idx = pages['start'] - 1
                end_idx = pages['end']

                print(f"\n--- Processing '{chunk_name}' (Pages {pages['start']} to {pages['end']}) ---")
                
                if start_idx >= len(pdf.pages):
                    print(f"Start page {pages['start']} is out of bounds. Skipping unit.")
                    continue
                
                unit_pages = pdf.pages[start_idx:end_idx]
                
                if not unit_pages:
                    print("No pages found in this range. Skipping.")
                    continue

                unit_text = extract_and_mark_page_text(unit_pages)
                
                if not unit_text.strip():
                    print("No text extracted from this unit. Skipping.")
                    continue
                
                print(f"  > Extracted {len(unit_text)} characters for this unit.")

                # --- NEW: Smart Batching Logic ---
                text_batches = []
                if len(unit_text) > MAX_CHARS_PER_CALL:
                    print(f"  > Unit text is too long. Splitting into smaller batches of ~{MAX_CHARS_PER_CALL} chars.")
                    # Create a list of text batches
                    for i in range(0, len(unit_text), MAX_CHARS_PER_CALL):
                        text_batches.append(unit_text[i:i + MAX_CHARS_PER_CALL])
                else:
                    # If the unit is small enough, it's a single batch
                    text_batches.append(unit_text)
                
                print(f"  > Processing unit in {len(text_batches)} batch(es).")
                # --- End of Smart Batching Logic ---

                # --- MODIFIED: Loop through the batches for the current unit ---
                for i, text_batch in enumerate(text_batches):
                    print(f"    - Sending Batch {i+1}/{len(text_batches)} to Gemini...")
                    
                    structured_data_str, error = get_structured_data_from_gemini(YOUR_API_KEY, text_batch)
                    
                    if error:
                        print(f"    - Error processing batch: {error}")
                        # Optional: Add a delay and retry logic here
                        time.sleep(5)
                        continue 

                    print("    - ...structured data received.")
                    
                    if structured_data_str:
                        if structured_data_str.startswith("```json"):
                            structured_data_str = structured_data_str[7:]
                        if structured_data_str.endswith("```"):
                            structured_data_str = structured_data_str[:-3]
                        
                        try:
                            parsed_data = json.loads(structured_data_str)
                            if isinstance(parsed_data, list):
                                all_structured_data.extend(parsed_data)
                                print(f"    - Successfully parsed and added {len(parsed_data)} records from batch.")
                            else:
                                print("    - Warning: API did not return a list. Response skipped.")
                        except json.JSONDecodeError:
                            print(f"    - CRITICAL: Could not decode JSON from API.")
                            print(f"    - Received response fragment: {structured_data_str[:500]}...")

    except Exception as e:
        print(f"A critical error occurred while opening or reading the PDF: {e}")
        return

    print(f"\n--- All units processed. Found a total of {len(all_structured_data)} sections. ---")
    
    final_data_with_ids = []
    for i, record in enumerate(all_structured_data):
        record['_id'] = f'rec_{i + 1}'
        final_data_with_ids.append(record)

    print(f"Saving final structured data to '{OUTPUT_JSON_PATH}'...")
    
    success, error = save_data_to_json(final_data_with_ids, OUTPUT_JSON_PATH)
    
    if error:
        print(error)
    else:
        print(f"\n✅ All done! Your structured content has been saved to {OUTPUT_JSON_PATH}")
        print("\n--- FINAL OUTPUT PREVIEW (first 2 items) ---")
        print(json.dumps(final_data_with_ids[:2], indent=2))
        print("-" * 40)
if __name__ == "__main__":
    main()