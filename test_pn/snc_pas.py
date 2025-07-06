import google.generativeai as genai
import pdfplumber
import os
import json
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio

# This function remains the same
def extract_text_from_pages(pages):
    """Extracts text from a list of pdfplumber page objects."""
    if not pages:
        return ""
    
    full_text = ""
    for page in pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + f"\n--- Page {page.page_number} ---\n"
    return full_text

# --- MODIFIED and IMPROVED extract_headers_and_text_with_gemini FUNCTION ---
def extract_headers_and_text_with_gemini(api_key, text_chunk, last_topic, current_page):
    """
    Sends a text chunk to the Gemini API to identify numbered headers and extract their corresponding text.
    Includes context from the previous chunk.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # --- NEW: Enhanced Prompt with Context ---
        prompt = f"""
        You are an expert assistant specializing in parsing mathematical and scientific textbooks.
        Your task is to meticulously scan the provided text and extract specific types of content: Definitions, Notations, Theorems, and Examples with their full solutions.

        The last known topic from the previous text chunk was: '{last_topic}'. Use this as a starting point for the context of the current text.

        You must identify and extract the following categories:
        1.  **Example**: A problem presented as an "Example".
        2.  **Solution**: A complete step-by-step solution immediately under an Example.

        For any mathematical formula or expression you find, write it in valid LaTeX code, enclosed by '$' for inline math and '$$' for block-level math.

        Format the output as a valid JSON list of objects. Each object must contain:
        1.  "id": A placeholder (e.g., "rec_temp").
        2.  "ex_name": The example preceded by the unit, like: "Unit 3 Example 2", "Unit 9 Example 10".
        3.  "ex_prob": The problem stated under the Example.
        4.  "solution": The full extracted solution, with all mathematical expressions converted to LaTeX.
        5.  "topic": The nearest preceding section header where the example was found.
        6.  "page_number": The page number where the content was found, as seen in the book.

        **Example of Desired Output:**
        ```json
        [
          {{
            "id": "rec_temp",
            "ex_name": "Unit 1 Example 3",
            "ex_prob": "Let ${{G_n}}_{{n=1}}^\\infty$ be a geometric sequence with common ratio $r$. Then the sum of the first $n$ terms $S_n$ is given by? ",
            "solution":"$$S_n = \\begin{{cases}} nG_1, & \\text{{if }} r=1 \\\\ G_1\\frac{{(1-r^n)}}{{1-r}} = G_1\\frac{{(r^n-1)}}{{r-1}}, & \\text{{if }} r \\neq 1. \\end{{cases}}$$",
            "topic": "Unit 1 Sequences and Series",
            "page_number": 1
          }}
        ]
        ```

        If no relevant content is found in this chunk, return an empty JSON list: [].

        Here is the text chunk to parse:
        {text_chunk}
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

# --- FULLY REVISED main() FUNCTION WITH OVERLAPPING BATCHES AND CONTEXT ---
def main():
    """Main function to run the book content extractor with overlapping page batches and context."""
    
    load_dotenv()
    YOUR_API_KEY = os.getenv("gemma_gemini_api") 
    INPUT_PDF_PATH = "Grade-12-Mathematics-Textbook.pdf"
    OUTPUT_JSON_PATH = "Grade_12_math_example_and_solution.json"
    
    # --- NEW: Define a batch and overlap size ---
    PAGES_PER_BATCH = 10  # Number of pages to process in each API call
    PAGE_OVERLAP = 2     # Number of pages to include from the previous batch

    PAGE_CHUNKS = {
        "Unit 1": {"start": 2, "end": 30},
        # Add other units here
    }

    if not YOUR_API_KEY:
        print("Error: Gemini API key not found.")
        return

    if not os.path.exists(INPUT_PDF_PATH):
        print(f"Error: PDF file not found at '{INPUT_PDF_PATH}'")
        return

    all_structured_data = []
    last_known_topic = "General Mathematics" # A default starting topic

    print("--- Starting PDF Processing with Overlapping Batches ---")
    try:
        with pdfplumber.open(INPUT_PDF_PATH) as pdf:
            num_total_pages = len(pdf.pages)
            print(f"Total pages in document: {num_total_pages}. Processing in batches of {PAGES_PER_BATCH} with {PAGE_OVERLAP} page overlap.")
            
            for chunk_name, pages in PAGE_CHUNKS.items():
                start_page = pages['start']
                end_page = pages['end']

                print(f"\n--- Processing '{chunk_name}' (Pages {start_page} to {end_page}) ---")
                
                if start_page >= num_total_pages:
                    print(f"Start page {start_page} is out of bounds. Skipping unit.")
                    continue
                
                # --- NEW: Loop with a step to create overlapping batches ---
                for i in range(start_page -1, end_page, PAGES_PER_BATCH - PAGE_OVERLAP):
                    batch_start = i
                    batch_end = min(i + PAGES_PER_BATCH, end_page)
                    
                    if batch_start >= batch_end:
                        continue
                        
                    print(f"  -> Processing batch: Pages {batch_start + 1} to {batch_end}")

                    page_batch = pdf.pages[batch_start:batch_end]
                    
                    if not page_batch:
                        continue

                    batch_text = extract_text_from_pages(page_batch)
                    
                    if not batch_text.strip():
                        print("     No text extracted from this batch. Skipping.")
                        continue
                    
                    print(f"     Extracted {len(batch_text)} characters. Sending to Gemini...")
                    
                    structured_data_str, error = extract_headers_and_text_with_gemini(
                        YOUR_API_KEY, batch_text, last_known_topic, batch_start + 1
                    )
                    
                    if error:
                        print(f"Error processing batch: {error}")
                        continue 

                    print("     ...structured data received.")
                    
                    if structured_data_str:
                        try:
                            parsed_data = json.loads(structured_data_str)
                            if isinstance(parsed_data, list) and parsed_data:
                                # --- NEW: Update context for the next iteration ---
                                last_found_topic_in_batch = parsed_data[-1].get("topic")
                                if last_found_topic_in_batch:
                                    last_known_topic = last_found_topic_in_batch
                                    
                                all_structured_data.extend(parsed_data)
                        except json.JSONDecodeError:
                            print(f"     Warning: Could not decode JSON from API. Response was: {structured_data_str}")

    except Exception as e:
        print(f"An error occurred while opening or reading the PDF: {e}")
        return

    print(f"\n--- All chunks processed. Found a total of {len(all_structured_data)} sections. ---")
    
    # Post-processing to remove duplicates can be added here if needed
    
    final_data_with_ids = []
    for i, record in enumerate(all_structured_data):
        record['id'] = f'rec_{i + 1}'
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