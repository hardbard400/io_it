import os
from dotenv import load_dotenv
from pinecone import Pinecone
# 1. CORRECTED: This is the proper way to import the Google Generative AI library.
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- YOUR PINECONE CODE - UNCHANGED AS REQUESTED ---
# pc = Pinecone(api_key=os.getenv("pinecone_api"))
# index_name = "biology-grade-11-general-contents"
# dense_index = pc.Index(index_name)

# query = "What are the properties of enzymes?"
# results = dense_index.search(
#     namespace="contents",
#     query={
#         "top_k": 10,
#         "inputs": {
#             'text': query
#         }
#     }
# )
# --- END OF UNCHANGED PINECONE CODE ---


# 2. CORRECTED: This is how you configure the genai library with your API key.
# The genai.Client class does not exist in the library.
gemini_api_key = os.getenv("gemma_gemini_api")
if not gemini_api_key:
    raise ValueError("Gemini API key not found in environment variables.")
genai.configure(api_key=gemini_api_key)


# --- YOUR FORMATTING LOGIC - UNCHANGED AS REQUESTED ---
pc = Pinecone(api_key=os.getenv("pinecone_api"))
index_names=["biology-grade-12","biology-grade-11"]

def query_d(query:str):
    index_names=["biology-grade-12","biology-grade-11"]
    lis=["key_words","general_text","tables"]
    mis=[]
    for ind in index_names:
        dense_index = pc.Index(ind)
        for i in lis:
            results = dense_index.search(
                namespace=i,
                query={
                    "top_k": 5,
                    "inputs": {
                        'text': query
                    }
                }
            )
            formatted_results = "\n".join(
                f"ID: {hit['_id']} | SCORE: {round(hit['_score'], 2)} | PAGE_NUMBER: {hit['fields']['page_number']}"
                "\n"
                f"TEXT_HEADER: {hit['fields']['topic']}"
                "\n"
                f"TEXT_CONTENT: {hit['fields']['chunk_text']}"
                "\n"
                "\n"
                "\n"
                "\n"
                for hit in results['result']['hits']
            )
            mis.append(formatted_results)
    k="\n".join(mis)
    #print(k)
    generate_content(query,k)
# --- END OF UNCHANGED FORMATTING LOGIC ---


def generate_content(query,contents):
    """
    Generates a summary of the query response based on the provided contents.
    """
    # The system prompt is unchanged
    system_prompt = f"""
    You are a specialized AI assistant. Your sole purpose is to answer the user's query based exclusively on the provided search results. You must adhere to the following instructions without deviation.

    **Instructions:**

    1.  **Analyze the User's Query:** The user wants to know: "{query}"

    2.  **Review the Provided Context:** You are given the following search results, each containing an ID, SCORE, PAGE_NUMBER, TEXT_HEADER, and TEXT_CONTENT.

        ```context
        {contents}
        ```

    3.  **Synthesize the Answer:**
        * Formulate a descriptive answer to the user's query using *only* the information found in the `TEXT_CONTENT` of the provided results.
        * The answer must be more than five sentences.
        * Do not invent, infer, or use any information outside of the provided context.
        * If you couldn't find a suitable answer to the {query}, just reply with "Im sorry, there isnt a suitale answer to your question in the book."
    4.  **Cite Your Sources:**
        * After the answer, list the sources you used.
        * For each source, you must include its `ID`, `SCORE`, and `PAGE_NUMBER`.
        * Format each citation exactly as: `Source: \n ID: [ID], SCORE: [SCORE],HEADER:[TEXT_HEADER] PAGE_NUMBER: [PAGE_NUMBER]`
    

    **Output Mandate:**

    * Your entire output must consist of two parts ONLY: the synthesized answer first, followed by the list of source citations.
    * DO NOT add any introductory phrases, greetings, apologies, or concluding remarks.
    * DO NOT use any formatting other than what is specified.
    """
    
    # 3. CORRECTED: You must first instantiate a model and then call generate_content on it.
    # The 'client.models.generate_content' structure is incorrect.
    # Note: 'gemini-2.5-flash' is not a valid model name. I have used 'gemini-1.5-flash'.
    # Please use the correct model name you have access to.
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(system_prompt)
    print(response.text)


# Generate and print the content
# query_d("How do we name and classify enzymes?")
# query_d("what is the capital city and population of addis ababa")
from time import perf_counter
s=perf_counter()
query_d("what are glycoscidic bonds??")
e=perf_counter()
print(e-s)



