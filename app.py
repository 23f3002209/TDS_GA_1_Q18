import os
import json
import numpy as np
import time
from flask import Flask, request, jsonify
from openai import OpenAI

# Initialize Flask App
app = Flask(__name__)

api_key = os.environ.get("OPENAI_API_KEY") 
client = OpenAI(
    api_key=api_key,
    base_url="https://aipipe.org/openai/v1" 
)

# --- NEW CODE (Add this instead) ---
def load_documents():
    try:
        with open('documents.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: documents.json not found. Run generate_data.py first!")
        return []

documents = load_documents()

# Store embeddings in memory so we don't compute them every time (Caching)
doc_embeddings = {}

def get_embedding(text):
    """
    Sends text to AI and gets back a list of numbers (vector).
    """
    response = client.embeddings.create(
        model="text-embedding-3-small", # Or your specific model
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(v1, v2):
    """
    Math formula to check how similar two vectors are.
    Returns a score between -1 and 1.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- PRE-COMPUTE STEP ---
# When the server starts, we turn all our documents into numbers immediately.
# This ensures the search is FAST (<200ms).
print("Pre-computing document embeddings... please wait.")
for doc in documents:
    doc_embeddings[doc['id']] = get_embedding(doc['text'])
print("Embeddings ready!")

def rerank_documents(query, candidates):
    """
    Takes the top candidates and asks the LLM to score them 0-10.
    """
    scored_results = []
    
    for doc in candidates:
        # We create a prompt for the AI
        prompt = f"""
        Query: "{query}"
        Document: "{doc['text']}"

        Rate the relevance of this document to the query on a scale of 0.0 to 10.0.
        Respond with ONLY the number.
        """
        
        try:
            # Ask the LLM
            response = client.chat.completions.create(
                model="gpt-4o-mini", # Use a fast/cheap model
                messages=[{"role": "user", "content": prompt}],
                temperature=0 # Keep it strict
            )
            
            # Extract the number
            content = response.choices[0].message.content.strip()
            score = float(content)
            
            # Normalize to 0-1 range (divide by 10)
            normalized_score = score / 10.0
            
            scored_results.append({
                "id": doc['id'],
                "score": normalized_score,
                "content": doc['text'],
                "metadata": {"source": "generated_db"}
            })
            
        except Exception as e:
            print(f"Error ranking doc {doc['id']}: {e}")
            # If AI fails, keep original score or default to 0
            scored_results.append({
                "id": doc['id'],
                "score": 0.0, 
                "content": doc['text']
            })

    # Sort by the new LLM score (Highest first)
    scored_results.sort(key=lambda x: x['score'], reverse=True)
    return scored_results

@app.route('/search', methods=['POST'])
def search():
    start_time = time.time()
    data = request.json
    
    user_query = data.get('query', '')
    top_k = data.get('k', 8)
    do_rerank = data.get('rerank', False)
    rerank_k = data.get('rerankK', 5)

    # 1. INITIAL RETRIEVAL (Vector Search)
    query_vector = get_embedding(user_query)
    
    initial_scores = []
    for doc in documents:
        doc_vector = doc_embeddings[doc['id']]
        score = cosine_similarity(query_vector, doc_vector)
        initial_scores.append({
            "id": doc['id'],
            "score": score,
            "content": doc['text'],
            "metadata": {"source": "database"}
        })
    
    # Sort by cosine similarity and take top K
    initial_scores.sort(key=lambda x: x['score'], reverse=True)
    top_candidates = initial_scores[:top_k]

    final_results = top_candidates
    is_reranked = False

    # 2. RE-RANKING (Optional Step)
    if do_rerank:
        # We only send the top candidates to the LLM to save money/time
        final_results = rerank_documents(user_query, top_candidates)
        # Trim to the requested rerank limit (e.g., top 5)
        final_results = final_results[:rerank_k]
        is_reranked = True

    # 3. CALCULATE METRICS
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)

    # 4. JSON RESPONSE
    response = {
        "results": final_results,
        "reranked": is_reranked,
        "metrics": {
            "latency": latency_ms,
            "totalDocs": len(documents)
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)