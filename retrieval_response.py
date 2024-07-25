import requests
import json
from sklearn.metrics.pairwise import cosine_similarity

api_key = "AIzaSyCzdCOyd-7os-SRgbEolxtwEEgYYkjKpsM"

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

    # Check the actual structure of the response and adjust accordingly
    try:
        generated_content = response_json["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError as e:
        raise ValueError(f"Unexpected response structure: {e}")

    return generated_content.strip()
