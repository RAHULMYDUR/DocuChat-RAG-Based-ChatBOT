import requests
import json
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_chunks(index, chunks, query, vectorizer, top_n=3):
    query_vec = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vec, top_n)
    return [chunks[i] for i in indices[0]]

def generate_response(retrieved_chunks, user_query, api_key):
    prompt = f'''
    You are a chatbot that answers questions based on the following documents:

    {retrieved_chunks}

    User Question: "{user_query}"

    Provide a coherent and contextually relevant answer.
    '''

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data), verify=False)
    response.raise_for_status()
    generated_content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return generated_content.strip()
