from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import json
import numpy as np
import faiss

api_key = "AIzaSyCzdCOyd-7os-SRgbEolxtwEEgYYkjKpsM"

def retrieve_relevant_chunks(index, chunks, query, vectorizer, top_n=3):
    query_vec = vectorizer.transform([query]).toarray()
    assert query_vec.shape[1] == index.d, f"Query vector dimension {query_vec.shape[1]} does not match index dimension {index.d}"
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
    
    # Print the response JSON for debugging
    response_json = response.json()
    print(json.dumps(response_json, indent=2))

    # Adjust based on actual response structure
    try:
        generated_content = response_json.get("choices", [{}])[0].get("text", "No content found")
    except KeyError as e:
        raise ValueError(f"Unexpected response structure: {e}")

    return generated_content.strip()
