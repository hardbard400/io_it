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
            # Adding the page number to the text provides better context for the LLM
            full_text += f"\n--- Page {page.page_number} ---\n" + page_text
    return full_text

# --- NEW: ASYNCHRONOUS version of the Gemini API call function ---
async def extract_headers_and_text_with_gemini_async(api_key, text_chunk, page_number, last_topic=""):
    """
    Sends a text chunk to the Gemini API asynchronously to extract structured data.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
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
        6.  "page_number": The page number where the content was found as seen in the book.

        **Example of Desired Output:**
        ```json
        [
          {{
            "id": "rec_temp",
            "ex_name": "Unit 1 Example 3",
            "ex_prob": "Let ${{G_n}}_{{n=1}}^\\infty$ be a geometric sequence with common ratio $r$. Then the sum of the first $n$ terms $S_n$ is given by? ",
            "solution":"$$S_n = \\begin{{cases}} nG_1, & \\text{{if }} r=1 \\\\ G_1\\frac{{(1-r^n)}}{{1-r}} = G_1\\frac{{(r^n-1)}}{{r-1}}, & \\text{{if }} r \\neq 1. \\end{{cases}}$$",
            "topic": "Unit 1 Sequences and Series",
            "page_number": {page_number}
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
        
        # Use the asynchronous version of the generate_content method
        response = await model.generate_content_async(prompt, safety_settings=safety_settings)
        
        cleaned_response = response.text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3].strip()
        
        return cleaned_response, None
        
    except Exception as e:
        return None, f"An error occurred with the Gemini API on page {page_number}: {e}"

# This function remains the same
def save_data_to_json(data_list, output_path):
    """Saves the list of dictionaries to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4)
        return True, None
    except Exception as e:
        return False, f"Error saving to JSON file: {e}"

# --- FULLY REVISED main() FUNCTION FOR ASYNCHRONOUS PROCESSING ---
async def main():
    """Main function to run the book content extractor asynchronously."""
    
    load_dotenv()
    YOUR_API_KEY = os.getenv("gemma_gemini_api") 
    INPUT_PDF_PATH = "Grade-12-Mathematics-Textbook.pdf"
    OUTPUT_JSON_PATH = "Grade_12_math_example_and_solution_async.json"
    
    # Using a larger batch size is more efficient with async
    PAGES_PER_BATCH = 8
    PAGE_OVERLAP = 3  

    PAGE_CHUNKS = {

 "Unit 1": {"start": 1, "end": 41}
#  "Unit 2": {"start": 47, "end": 103},
#  "Unit 3": {"start": 109, "end": 160},
#  "Unit 4": {"start": 167, "end": 209},
#  "Unit 5": {"start": 213, "end": 261},
#  "Unit 6": {"start": 277, "end": 297},
#  "Unit 7": {"start": 302, "end": 322},
#  "Unit 8": {"start": 328, "end": 371},
#  "Unit 9": {"start": 376, "end": 394},

}

    if not YOUR_API_KEY:
        print("Error: Gemini API key not found.")
        return

    if not os.path.exists(INPUT_PDF_PATH):
        print(f"Error: PDF file not found at '{INPUT_PDF_PATH}'")
        return

    all_structured_data = []
    
    print("--- Starting Asynchronous PDF Processing ---")
    try:
        with pdfplumber.open(INPUT_PDF_PATH) as pdf:
            tasks = []
            last_known_topic = "General Mathematics"

            for chunk_name, pages_info in PAGE_CHUNKS.items():
                start_page = pages_info['start']
                end_page = pages_info['end']

                print(f"\n--- Preparing tasks for '{chunk_name}' (Pages {start_page} to {end_page}) ---")
                
                for i in range(start_page - 1, end_page, PAGES_PER_BATCH - PAGE_OVERLAP):
                    batch_start = i
                    batch_end = min(i + PAGES_PER_BATCH, end_page)
                    
                    if batch_start >= batch_end:
                        continue

                    page_batch = pdf.pages[batch_start:batch_end]
                    batch_text = extract_text_from_pages(page_batch)
                    
                    if not batch_text.strip():
                        continue
                    
                    # Create a task for each API call and add it to the list
                    task = extract_headers_and_text_with_gemini_async(
                        YOUR_API_KEY, batch_text, batch_start + 1, last_known_topic
                    )
                    tasks.append(task)
            
            print(f"\n--- Running {len(tasks)} tasks concurrently ---")
            # Run all the created tasks at the same time
            results = await asyncio.gather(*tasks)
            print("--- All tasks completed ---")

            for structured_data_str, error in results:
                if error:
                    print(f"  -> Error processing a batch: {error}")
                    continue
                
                if structured_data_str:
                    try:
                        parsed_data = json.loads(structured_data_str)
                        if isinstance(parsed_data, list):
                            all_structured_data.extend(parsed_data)
                    except json.JSONDecodeError:
                        print(f"  -> Warning: Could not decode JSON from API. Response was: {structured_data_str}")
    
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        return

    print(f"\n--- All chunks processed. Found a total of {len(all_structured_data)} sections. ---")
    
    # Re-apply sequential IDs after all data is collected
    final_data_with_ids = []
    # Sort data by page number to maintain order
    all_structured_data.sort(key=lambda x: x.get('page_number', 0))
    for i, record in enumerate(all_structured_data):
        record['id'] = f'rec_{i + 1}'
        final_data_with_ids.append(record)

    print(f"Saving final structured data to '{OUTPUT_JSON_PATH}'...")
    
    success, error = save_data_to_json(final_data_with_ids, OUTPUT_JSON_PATH)
    
    if error:
        print(error)
    else:
        print(f"\n✅ All done! Your structured content has been saved to {OUTPUT_JSON_PATH}")
        print("\n--- FINAL OUTPUT PREVIEW (first 2 items) ---")
        print(json.dumps(final_data_with_ids[:2], indent=4))
        print("-" * 35)

if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function
    asyncio.run(main())