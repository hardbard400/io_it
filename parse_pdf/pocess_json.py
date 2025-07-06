import json
import re

def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes scientific text to improve semantic meaning for
    vector search embeddings. This version includes all identified Unicode chars.
    """
    if not isinstance(text, str):
        return text

    # Step 1: Handle a comprehensive list of Unicode and special character replacements
    replacements = {
        # Punctuation and Quotes
        '\u2019': "'",  # Right Single Quote
        '\u2018': "'",  # Left Single Quote
        '\u201c': '"',  # Left Double Quote
        '\u201d': '"',  # Right Double Quote
        '\u2013': "-",  # En Dash
        '\u2026': "...",  # Ellipsis
        
        # Mathematical and Scientific Symbols
        '\u2192': ' yields ',      # Right Arrow
        '\u279e': ' yields ',      # Another Right Arrow
        '\u2190': ' from ',        # Left Arrow
        '\u00b0': ' degrees ',     # Degree Sign
        '\u03b4': 'delta',         # Greek Delta
        '\u03bc': 'um',            # Greek Mu (for micrometers)
        '\u03a8': 'Psi',           # Greek Psi
        '\u221d': ' is proportional to ', # Proportional To
        '\u00d7': ' times ',       # Multiplication Sign
        '\u00f7': ' divided by ',  # Division Sign
        
        # Formatting
        '\u2022': ' ',             # Bullet Point
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Step 2: Normalize whitespace and line breaks BEFORE formula reconstruction
    # This makes regex for formulas simpler and more reliable
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text) # Collapse multiple spaces into one

    # Step 3: Reconstruct chemical formulas like C H O 6 12 6 -> C6H12O6
    # This regex repeatedly finds a letter-group followed by a space and a number-group.
    # We loop it to catch all occurrences in complex formulas like 'C H O 6 12 6'.
    for _ in range(5):
        text = re.sub(r'([A-Za-z])\s+(\d+)', r'\1\2', text)
        
    # A more specific pattern to catch the common 'C H O' formula structure if needed,
    # though the general loop above is often sufficient.
    text = re.sub(r'([C|c])([H|h])([O|o])(\d+)(\d+)(\d+)', r'\1\4\2\5\3\6', text)

    # Final cleanup of any remaining excess whitespace
    text = text.strip()

    return text

# --- Main script execution ---

# Load your structured content file
try:
    with open('structured_biology_content_2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: The file 'structured_biology_content_2.json' was not found.")
    data = []

# Create a new list for the processed data
processed_data = []

# Process each record
for record in data:
    # Make a copy to avoid modifying the original list in memory
    new_record = record.copy()
    if 'chunk_text' in new_record:
        # Apply the comprehensive preprocessing function
        new_record['chunk_text'] = preprocess_text(new_record['chunk_text'])
    processed_data.append(new_record)

# Save the newly formatted data to a new file
output_filename = 'structured_biology_content_2_processed.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"Processing complete. The thoroughly formatted data has been saved to '{output_filename}'")

# --- Verification: Before and After ---
# Pick a record with varied symbols, like rec_12, which includes °C
if len(data) > 11 and len(processed_data) > 11:
    print("\n--- Verification: Before and After (rec_12) ---")
    print("\nOriginal Text:")
    print(data[11]['chunk_text'])
    print("\nProcessed Text:")
    print(processed_data[11]['chunk_text'])