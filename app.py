import streamlit as st
from file_handler import extract_text_from_pdf
from processing import chunk_documents, vectorize_chunks, store_vectors_in_faiss
from retrieval_response import retrieve_relevant_chunks, generate_response

# Define your API key here
api_key = "AIzaSyCzdCOyd-7os-SRgbEolxtwEEgYYkjKpsM"

# App title and configuration
st.set_page_config(page_title="RAG-based Chatbot")

# Sidebar for file upload
with st.sidebar:
    st.title("Upload File")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How may I assist you today?"}]
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# Function to get user input
def get_text():
    return st.text_input("Type your question here:", "", key="input")

# Function to generate response
def generate_response(prompt, api_key):
    # Generate a response using the API
    # Replace this with your actual response generation logic
    response = f"Response to: {prompt}"
    return response

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input and response
with st.container():
    user_input = get_text()
    if st.button("Get Answer") and user_input:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Generate response
        response = generate_response(user_input, api_key)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Clear user input
        st.session_state.user_query = ""

# Display new messages
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
