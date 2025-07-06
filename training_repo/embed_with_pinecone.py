def embed_with_pinecone():
        from dotenv import load_dotenv
        import os
        import json
        import random
        import string
        from google import genai
        import time
        import gemini_record
        from google import genai
        load_dotenv()
        pinecone_api=os.getenv("pinecone_api")
        gemini_api_key=os.getenv("gemma_gemini_api")

        def generate_random_string(length=14):
            characters = string.ascii_letters + string.digits + "-"
            return ''.join(random.choices(characters, k=length))

        """#create a pinecone index

        """

        # Import the Pinecone library
        from pinecone import Pinecone

        # Initialize a Pinecone client with your API key
        pc = Pinecone(api_key=pinecone_api)

        # Generate random index name and namespace name
        index_name = generate_random_string()
        namespace_name = generate_random_string()

        # Create a dense index with integrated embedding
        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model":"llama-text-embed-v2",
                    "field_map":{"text": "chunk_text"}
                }
            )

        records= gemini_record.gemini_segement_sentences()
        dense_index = pc.Index(index_name)
        dense_index.upsert_records(namespace_name, records)
        time.sleep(10)

        

        gemini_client = genai.Client(api_key=gemini_api_key)

        ##########################--------------------------##############################
        query = "does ac solve the problem of entropy reversal"
        reranked_results = dense_index.search(
            namespace="query-assimov-namespace",
            query={
                "top_k": 10,
                "inputs": {
                    'text': query
                }
            },
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 10,
                "rank_fields": ["chunk_text"]
            }
        )
        hit_texts = []
        for hit in reranked_results['result']['hits']:
            text = hit['fields'].get('chunk_text', '')
            hit_texts.append(text)
            print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {text:<50}")

        # === Insert summarization using Gemini ===
        # You may choose to limit number of hits if needed
        combined_text = "\n".join(f"- {t}" for t in hit_texts)

        system_prompt = (
            "this is the initial query :{}, and this is the  collection of close answers:{}"
            "if you find that the answers you get dont match with the question asked, just answer on your own"
            "Summarize the following search results in a concise paragraph:\n"
            "if the contents of {} dont contain sufficient answers for {} then answer with, 'not enough data for satisfactory answer'"
            "Only summarize the parts relevant, get straight to the summarization, no need to refer to the structure"
            "answer in a maximum of 3 sentences"
            "make no mention of the search process or the findings process".format(query, combined_text,query,combined_text)
            # "Briefly state the overall finding."
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=[system_prompt]
        )
        return response.txt


