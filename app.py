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

# HTML and CSS for fixed input and chat history
st.markdown("""
    <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 80vh;
            overflow: hidden;
        }
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        .chat-input {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: white;
            padding: 10px;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        }
    </style>
    <div class="chat-container">
        <div class="chat-history">
            <!-- Chat history will be displayed here -->
        </div>
        <div class="chat-input">
            <!-- Input field and button will be here -->
        </div>
    </div>
""", unsafe_allow_html=True)

# Display chat messages in chat history
chat_history_html = ""
for message in st.session_state.chat_history:
    role_class = "user" if message["role"] == "user" else "assistant"
    chat_history_html += f'<div class="{role_class}">{message["content"]}</div>'
st.markdown(f'<div class="chat-history">{chat_history_html}</div>', unsafe_allow_html=True)

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
    chat_history_html = ""
    for message in st.session_state.chat_history:
        role_class = "user" if message["role"] == "user" else "assistant"
        chat_history_html += f'<div class="{role_class}">{message["content"]}</div>'
    st.markdown(f'<div class="chat-history">{chat_history_html}</div>', unsafe_allow_html=True)
