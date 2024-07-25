import streamlit as st

# Try to import functions from retrieval_response
try:
    from retrieval_response import retrieve_relevant_chunks, generate_response
except ImportError as e:
    st.error(f"ImportError: {e}")
    st.stop()

from file_handler import extract_text_from_pdf
from processing import chunk_documents, vectorize_chunks, store_vectors_in_faiss

# Define your API key here
api_key = "AIzaSyCzdCOyd-7os-SRgbEolxtwEEgYYkjKpsM"

st.title("DocuChat - RAG based Chatbot")

def main():
    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {"question": "", "answer": "Hi, I am your RAG-Based ChatBOT. Please upload the PDF file if you didn't."}
        ]

    if uploaded_file is not None:
        st.sidebar.write(f"File uploaded: {uploaded_file.name}")

        # Read the PDF file
        documents = extract_text_from_pdf(uploaded_file)

        # Convert documents to chunks and vectors
        chunks = chunk_documents(documents)
        vectors, vectorizer = vectorize_chunks(chunks)

        # Store vectors in FAISS
        index = store_vectors_in_faiss(vectors)
        
        # Store the index and vectorizer in session state for later use
        st.session_state.index = index
        st.session_state.vectorizer = vectorizer
        st.session_state.chunks = chunks

    user_query = st.chat_input("Ask a question:")
    if user_query:
        if 'index' in st.session_state and 'vectorizer' in st.session_state and 'chunks' in st.session_state:
            try:
                retrieved_chunks = retrieve_relevant_chunks(st.session_state.index, st.session_state.chunks, user_query, st.session_state.vectorizer)
                response = generate_response("\n\n".join(retrieved_chunks), user_query, api_key)
            except Exception as e:
                response = f"An error occurred while generating the response: {str(e)}"
        else:
            response = "Please upload a PDF file first."

        # Append the question and answer to the chat history
        st.session_state.chat_history.append({"question": user_query, "answer": response})

    # Display the chat history
    for chat in st.session_state.chat_history:
        if chat['question']:
            with st.chat_message("user"):
                st.write(chat['question'])
        with st.chat_message("assistant"):
            st.write(chat['answer'])

if __name__ == "__main__":
    main()
