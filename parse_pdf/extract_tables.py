import google.generativeai as genai
import pdfplumber
import os
import json
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# This function remains the same
def extract_text_from_pages(pages):
    """Extracts text from a list of pdfplumber page objects."""
    if not pages:
        return ""
    
    full_text = ""
    for page in pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

# This function and its detailed prompt remain the same
def extract_headers_and_text_with_gemini(api_key, text_chunk):
    """
    Sends a text chunk to the Gemini API to identify numbered headers and extract their corresponding text.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        You are an expert data synthesizer and technical writer. Your primary task is to identify tables within the provided text from a textbook, comprehend the information and relationships they contain, and then rewrite that information as a dense, coherent, and self-contained paragraph.

        **Instructions:**
        1.  **Identify Tables:** Scan the text to locate any content that is clearly a table. Tables are preceded by a label like "Table 4.1" and have a structured layout of rows and columns.
        2.  **If you do not find the a title, but if you find a table, you can assume that the table is continuing from the page behind it or previous to it.
        3.  **The tables title should not be anything like a figure or something else or activity.The title should always be the word "Table" and then an integer like "Table 1.1" or "Table 2.2" or Table is "3.3".But they should not contain words like "Activity" or "Figure". Like "Activity 2.3" or "Figure 1.2" . It should always be "Table" and then an integer.
        2.  **Comprehend and Synthesize:** Do not just extract the text. You must understand the relationships between the columns and rows to describe how different items compare or relate to each other.
        3.  **Summarize into Prose:** For each table you find, write a single, fluid paragraph that summarizes its entire contents.
            * Begin by stating the purpose or title of the table.
            * Logically connect the data points. For example, instead of listing "Process: Simple Diffusion" and "Requirement for ATP?: No", you should state, "Simple diffusion is a process that does not require ATP."
            * **Crucial Constraint:** Do not add any information that is not explicitly present in the table. Do not omit any details from the table. The goal is a faithful transformation of data from a tabular format to a paragraph format.
        4.  **Strict Exclusion:** Ignore all content that is not part of a clearly identifiable table. Do not process regular text, activities, review questions, or figures.

        **Output Format:**
        * Your output **MUST** be a single, valid JSON list of objects.
        * Each object in the list represents one table and **MUST** contain exactly three key-value pairs:
            1.  `"table_name"`: The full name of the table (e.g., "Table 4.5 The transport processes compared").
            2.  `"chunk_text"`: The complete summary paragraph you created for that table.
            3.  `"page_number"`: the page number where the begining of the table was found, formatted as an integer.

        **Example of Desired Output:**
        ```json
        [
          {{
            "table_name": "Table 4.5 The transport processes compared",
            "chunk_text": "This table compares six different transport processes based on concentration influence, ATP requirement, the type of particles moved, and whether transport proteins are needed. Simple diffusion, facilitated diffusion, and osmosis all occur from a high to a low concentration or water potential and do not require ATP. Simple diffusion moves small, lipid-soluble, non-polar particles without proteins. Facilitated diffusion moves ions and medium-sized particles with the help of proteins. Osmosis specifically moves water molecules and does not require transport proteins. In contrast, active transport, endocytosis, and exocytosis all require ATP. Active transport moves various particles from a low to high concentration using transport proteins. Both endocytosis and exocytosis can occur in either direction to move very large particles and also rely on transport proteins."
            "page_number" 1
          }},
          {{
            "table_name": "Table 4.1 Key events in cell biology",
            "chunk_text": "This table lists key events in the study of cell biology chronologically. The timeline begins in 1595 with Janssen building the first compound microscope and continues with Redi's 1626 postulation against spontaneous generation. Other key events include Hooke's 1665 discovery of 'cells' in cork, Leeuwenhoek's 1674 discovery of protozoa, and Brown's 1833 description of the cell nucleus. The cell theory was proposed by Schleiden and Schwann in 1839. Later discoveries include Kölliker describing mitochondria (1857), Virchow's statement 'omnis cellula e cellula' (1858), Miescher's discovery of DNA (1869), and Flemming's description of chromosome behaviour (1879). The sequence concludes with Golgi describing the Golgi apparatus (1898), the invention of the transmission electron microscope (1939), Watson and Crick's double-helix model (1953), the first scanning electron microscope (1965), and the draft of the human genome sequence in 2000."
            "page_number" 9
          }}
        ]
        ```

        If the text chunk contains no tables that match these rules, return an empty JSON list: `[]`.

        **Here is the text chunk to analyze:**
        ---
        {text_chunk}
        ---
        """
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        if response.candidates and response.candidates[0].finish_reason.name != "STOP":
            print(f"Warning: Content generation stopped for reason: {response.candidates[0].finish_reason.name}")
            if response.prompt_feedback.block_reason:
                return None, f"API call blocked by safety settings: {response.prompt_feedback.block_reason.name}"

        cleaned_response = response.text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3].strip()
        
        return cleaned_response, None
    except Exception as e:
        return None, f"An error occurred with the Gemini API: {e}"

# This function remains the same
def save_data_to_json(data_list, output_path):
    """Saves the list of dictionaries to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4)
        return True, None
    except Exception as e:
        return False, f"Error saving to JSON file: {e}"


# --- MODIFIED main() FUNCTION ---
# --- FULLY REVISED main() FUNCTION WITH BATCHING ---

def main():
    """Main function to run the book content extractor using a dictionary and batch processing."""
    
    load_dotenv()
    YOUR_API_KEY = os.getenv("gemma_gemini_api") 
    INPUT_PDF_PATH = "/workspaces/io_it/pdf's/Grade-9-Biology-Textbook.pdf"
    OUTPUT_JSON_PATH = "Grade_9_structured_biology_table.json"
    
    # --- NEW: Define a batch size ---
    # This is the number of pages to process in each API call. Tune if needed.
    BATCH_SIZE = 10

    PAGE_CHUNKS = {
        "Unit 1": {"start": 0, "end":13 },
        "Unit 2": {"start":17 , "end":51},
        "Unit 3": {"start":54, "end":121},
        "Unit 4": {"start":127, "end":169},
        "Unit 5": {"start":175, "end":200},  
        "Unit 6": {"start":204, "end":228}
        
         
    }

    if not YOUR_API_KEY:
        print("Error: Gemini API key not found.")
        return

    if not os.path.exists(INPUT_PDF_PATH):
        print(f"Error: PDF file not found at '{INPUT_PDF_PATH}'")
        return

    all_structured_data = []

    print("--- Starting PDF Processing by Defined Chunks (in Batches) ---")
    try:
        with pdfplumber.open(INPUT_PDF_PATH) as pdf:
            num_total_pages = len(pdf.pages)
            print(f"Total pages in document: {num_total_pages}. Processing in batches of {BATCH_SIZE} pages.")
            
            # Loop over the main Units
            for chunk_name, pages in PAGE_CHUNKS.items():
                start_page = pages['start']
                end_page = pages['end']

                print(f"\n--- Processing '{chunk_name}' (Pages {start_page + 1} to {end_page}) ---")
                
                if start_page >= num_total_pages:
                    print(f"Start page {start_page + 1} is out of bounds. Skipping unit.")
                    continue
                
                # --- NEW: Loop over the unit in smaller batches ---
                for i in range(start_page, end_page, BATCH_SIZE):
                    batch_start = i
                    batch_end = min(i + BATCH_SIZE, end_page) # Ensure we don't go past the unit's end page
                    
                    print(f"  -> Processing batch: Pages {batch_start + 1} to {batch_end}")

                    page_batch = pdf.pages[batch_start:batch_end]
                    
                    if not page_batch:
                        continue

                    batch_text = extract_text_from_pages(page_batch)
                    
                    if not batch_text.strip():
                        print("No text extracted from this batch. Skipping.")
                        continue
                    
                    print(f"     Extracted {len(batch_text)} characters. Sending to Gemini...")
                    
                    structured_data_str, error = extract_headers_and_text_with_gemini(YOUR_API_KEY, batch_text)
                    
                    if error:
                        print(f"     Error processing batch: {error}")
                        continue 

                    print("     ...structured data received.")
                    
                    if structured_data_str:
                        try:
                            parsed_data = json.loads(structured_data_str)
                            if isinstance(parsed_data, list):
                                all_structured_data.extend(parsed_data)
                            else:
                                print("     Warning: API did not return a list. Response skipped.")
                        except json.JSONDecodeError:
                            print(f"     Warning: Could not decode JSON from API. Response was: {structured_data_str}")

    except Exception as e:
        print(f"An error occurred while opening or reading the PDF: {e}")
        return

    print(f"\n--- All chunks processed. Found a total of {len(all_structured_data)} sections. ---")
    
    # --- IMPORTANT: Re-apply sequential IDs after all data is collected ---
    final_data_with_ids = []
    for i, record in enumerate(all_structured_data):
        record['_id'] = f'rec_{i + 1}'
        final_data_with_ids.append(record)

    print(f"Saving final structured data to '{OUTPUT_JSON_PATH}'...")
    
    success, error = save_data_to_json(final_data_with_ids, OUTPUT_JSON_PATH)
    
    if error:
        print(error)
    else:
        print(f"\nAll done! Your structured content has been saved to {OUTPUT_JSON_PATH}")
        print("\n--- FINAL OUTPUT PREVIEW (first 2 items) ---")
        print(json.dumps(final_data_with_ids[:2], indent=4))
        print("-" * 35)


if __name__ == "__main__":
    main()

