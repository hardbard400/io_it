import PyPDF2
import re
import os

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF file.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        str: The extracted text from the PDF, or None on failure.
    """
    print(f"Attempting to open and read PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"Error: The file was not found at {pdf_path}")
        return None

    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"PDF has {len(pdf_reader.pages)} pages.")
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Attempt to extract text and default to an empty string if None
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                except Exception as e:
                    print(f"Could not extract text from page {page_num + 1}: {e}")
        print("Successfully extracted text from PDF.")
        return text
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None

def chunk_text_by_paragraph(text, sentences_per_chunk, overlap_sentences):
    """
    Chunks text by paragraphs with a specified number of sentences and overlap.
    This approach preserves semantic context by keeping sentences intact.

    Args:
        text (str): The input text to be chunked.
        sentences_per_chunk (int): The desired number of sentences in each chunk.
        overlap_sentences (int): The number of sentences to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        print("Warning: Input text is empty. Returning no chunks.")
        return []
    
    # A more robust sentence splitting regex. It handles various sentence endings
    # and avoids splitting on abbreviations like "Mr." or "U.S.".
    # For production use, a library like NLTK or spaCy is recommended for higher accuracy.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    sentences = [s.strip().replace('\n', ' ') for s in sentences if s.strip()]

    if not sentences:
        print("Warning: No sentences were extracted from the text.")
        return []

    print(f"Chunking {len(sentences)} sentences into chunks of {sentences_per_chunk} with {overlap_sentences} sentence overlap.")

    # Validate chunking parameters
    if sentences_per_chunk <= 0:
        print("Error: sentences_per_chunk must be positive.")
        return []
    if overlap_sentences >= sentences_per_chunk:
        print("Error: overlap_sentences must be less than sentences_per_chunk.")
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        # Define the end of the current chunk
        end = i + sentences_per_chunk
        
        # Create the chunk by joining the sentences
        chunk = " ".join(sentences[i:end])
        chunks.append(chunk)
        
        # Determine the starting point of the next chunk
        next_start = i + sentences_per_chunk - overlap_sentences
        
        # Break if the next start is the same as the current one to prevent infinite loops
        if next_start <= i:
            break
        i = next_start

    print(f"Generated {len(chunks)} paragraph chunks.")
    return chunks


if __name__ == '__main__':
    # --- Configuration ---
    # Replace with the path to your PDF file.
    PDF_FILE_PATH = "Grade-11-Biology-Textbook.pdf" 
    
    # --- Create a dummy PDF for testing if it doesn't exist ---
    if not os.path.exists(PDF_FILE_PATH):
        print(f"'{PDF_FILE_PATH}' not found. Creating a dummy PDF for demonstration.")
        try:
            # Using reportlab to create a sample PDF
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph

            c = canvas.Canvas(PDF_FILE_PATH, pagesize=letter)
            styles = getSampleStyleSheet()
            style = styles['Normal']
            text_content = """The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation and they were given millions of dollars to make this vision come true. Ultimately, it was discovered that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British Governments cut off all undirected, exploratory research in AI. The first AI winter began. In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research. However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer-lasting AI winter began."""
            
            p = Paragraph(text_content, style)
            p.wrapOn(c, 500, 700)
            p.drawOn(c, 50, 600)
            c.save()
            print(f"Dummy PDF '{PDF_FILE_PATH}' created.")
        except ImportError:
            print("Please install reportlab (`pip install reportlab`) to create a dummy PDF.")
        except Exception as e:
            print(f"Failed to create dummy PDF: {e}")

    # --- Processing ---
    extracted_text = extract_text_from_pdf(PDF_FILE_PATH)

    if extracted_text:
        # Chunk the text using the sentence-based paragraph method
        # You can adjust these parameters for your specific needs
        SENTENCES_PER_CHUNK = 5
        OVERLAP_SENTENCES = 1
        
        paragraph_chunks = chunk_text_by_paragraph(
            extracted_text, 
            sentences_per_chunk=SENTENCES_PER_CHUNK, 
            overlap_sentences=OVERLAP_SENTENCES
        )
        
        # --- Output ---
        # Print the first 3 chunks as an example
        print(f"\n--- Displaying the first 3 (out of {len(paragraph_chunks)}) generated chunks ---")
        for i, chunk in enumerate(paragraph_chunks[:3]):
            print(f"\n[Chunk {i+1}]")
            print(chunk)
            print("-" * 25)
            
        # You can now take the 'paragraph_chunks' list and use it for your embedding process.
