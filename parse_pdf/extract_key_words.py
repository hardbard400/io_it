import google.generativeai as genai
import pdfplumber
import os
from dotenv import load_dotenv
import json # <-- 1. IMPORTED for robust JSON handling

def extract_text_from_pages(pages):
    """Extracts text from a list of pdfplumber page objects."""
    if not pages:
        return ""
    
    full_text = ""
    for page in pages:
        page_text = page.extract_text()
        if page_text:
            # Adding the page number to the text itself for better context
            full_text += f"--- START OF PAGE {page.page_number} ---\n"
            full_text += page_text + "\n"
            full_text += f"--- END OF PAGE {page.page_number} ---\n\n"
    return full_text

def get_definitions_with_gemini(api_key, text_chunk):
    """Sends a text chunk to the Gemini API to find and format definitions."""
    try:
        genai.configure(api_key=api_key)
        # Using a model that is optimized for JSON output can improve reliability
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Updated prompt to explicitly ask for a JSON list
        prompt = f"""
        You are a helpful assistant that parses academic textbooks.
        Your task is to look through the provided text and find all terms listed ONLY under a "KEY WORDS"section.
        For each key word found, you must extract its definition exactly as provided in the text. Do not add or remove any information.
        
        You will format the output as a valid JSON list of objects. Each object must contain four key-value pairs:
        1. "id": goes sequentially eg, "rec_1","rec_2","rec_3","rec_4","rec_5","rec_6","rec_7","rec_8".
        2. "chunk text":The definition of the word as given in the text. it should have this format: "chunk text": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll." where photo synthesis is the keyword from the book and the definition is also the one from the book, but you should format it so it says "keyword means...." or "keyword is ....",
        3. "topic": should be the nearest header you find that encloses the keywords context. the header should not be the following: "UNIT 1", "Activity", "KEY WORDS", "Contents", "Learning competencies", or "Figure". headers should look like :that start with a number, a period, and another number, like "1.1", "1.2", "1.3", etc. and headers that ask questions like "What is science?" or "what is the scientific Method?" and other headers that dont seem to be activities or figures[images] but rather descriptive statements like "validity" or "The methods of science" and dont extract text where its startes with "by the end of this section....", and it goes on to state the objective.
        4. "page_number": the page number where the key word and definition were found, formatted as an integer.
         **Example of Desired Output:**
        ```json
        [
          {{
              
        "id": "rec_1",
        "chunk text": "micro-organism is a very small organism, usually having just one cell.",
        "topic": "1.1 Bacteria",
        "page_number": 4
    
          }}
        ]
        ```
        If you find multiple key words, include all of them in the JSON list.
        If no key words or definitions are found in this chunk, return an empty JSON list: []

        Here is the text chunk:
        {text_chunk}
        """

        response = model.generate_content(prompt)
        
        # Clean up the response to ensure it's valid JSON
        # Models can sometimes wrap the JSON in markdown backticks
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        return cleaned_response, None
    except Exception as e:
        return None, f"Error with Gemini API: {e}"

def save_definitions_to_json(definitions_list, output_path):
    """Saves the list of definition dictionaries to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Use json.dump for proper JSON formatting and writing
            json.dump(definitions_list, f, indent=4)
        return True, None
    except Exception as e:
        return False, f"Error saving to JSON file: {e}"

# --- 4. NEW HELPER FUNCTION TO SORT THE FINAL LIST ---
def sort_definitions_by_page(definitions_list):
    """
    Sorts a list of definition dictionaries in ascending order based on page number.
    
    Args:
        definitions_list: A list of dictionaries, where each dict is expected
                          to have a 'page_number' key.
                          
    Returns:
        A new list containing the sorted dictionaries.
    """
    # Use .get() to avoid errors if a dictionary is missing the 'page_number' key
    return sorted(definitions_list, key=lambda item: item.get('page_number', 0))

def main():
    """Main function to run the book keyword extractor with chunking."""
    
    # --- IMPORTANT: CONFIGURE THESE VALUES ---
    load_dotenv()
    YOUR_API_KEY = os.getenv("gemma_gemini_api") # Replace with your actual Gemini API key
    INPUT_PDF_PATH = "/workspaces/io_it/pdf's/Grade-9-Biology-Textbook.pdf"
    OUTPUT_JSON_PATH = "Grade_9_Biology_keyword_definitions.json" # Saving as .json is more appropriate
    
    PAGES_TO_PROCESS = 227
    CHUNK_SIZE = 10
    # -----------------------------------------

    if not os.path.exists(INPUT_PDF_PATH):
        print(f"Error: PDF file not found at '{INPUT_PDF_PATH}'")
        return

    # This list will store Python dictionaries from all chunks
    all_definitions = []

    print("--- Starting PDF Processing ---")
    try:
        with pdfplumber.open(INPUT_PDF_PATH) as pdf:
            num_total_pages = len(pdf.pages)
            pages_to_process = num_total_pages
            if PAGES_TO_PROCESS and 0 < PAGES_TO_PROCESS < num_total_pages:
                pages_to_process = PAGES_TO_PROCESS

            print(f"Total pages in document: {num_total_pages}. Processing up to page {pages_to_process}.")
            print(f"Processing in chunks of {CHUNK_SIZE} pages.")

            for start_page in range(0, pages_to_process, CHUNK_SIZE):
                end_page = min(start_page + CHUNK_SIZE, pages_to_process)
                print(f"\nStep 1: Processing pages {start_page + 1} to {end_page}...")

                page_chunk = pdf.pages[start_page:end_page]
                chunk_text = extract_text_from_pages(page_chunk)
                
                if not chunk_text:
                    print("No text extracted from this chunk. Skipping.")
                    continue
                
                print(f"Successfully extracted {len(chunk_text)} characters from chunk.")
                print("Step 2: Sending text chunk to Gemini for analysis...")
                
                definitions_str, error = get_definitions_with_gemini(YOUR_API_KEY, chunk_text)
                
                if error:
                    print(error)
                    continue 

                print("...definitions received from chunk.")
                
                # --- 2. PARSE AND MERGE JSON DATA ---
                if definitions_str:
                    try:
                        # Convert the JSON string from the API into a Python list
                        parsed_definitions = json.loads(definitions_str)
                        # Ensure it's a list before extending
                        if isinstance(parsed_definitions, list):
                            # Use extend to add all items from the new list to our master list
                            all_definitions.extend(parsed_definitions)
                        else:
                            print("Warning: API did not return a list. Response skipped.")
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from the API response. Response was: {definitions_str}")

    except Exception as e:
        print(f"An error occurred while opening or reading the PDF: {e}")
        return

    print("\n--- All chunks processed. Aggregating and sorting results. ---")

    # --- 5. SORT THE FINAL LIST BEFORE SAVING ---
    print(f"Found a total of {len(all_definitions)} definitions. Sorting by page number...")
    sorted_definitions = sort_definitions_by_page(all_definitions)

    print(f"\nStep 3: Saving final list of definitions to '{OUTPUT_JSON_PATH}'...")
    success, error = save_definitions_to_json(sorted_definitions, OUTPUT_JSON_PATH)
    
    if error:
        print(error)
    else:
        print(f"\nAll done! Your extracted definitions have been saved to {OUTPUT_JSON_PATH}")
        print("\n--- FINAL OUTPUT PREVIEW (first 5 items) ---")
        # Preview the first 5 entries from the sorted list
        print(json.dumps(sorted_definitions[:5], indent=4))
        print("-" * 35)


if __name__ == "__main__":
    main()