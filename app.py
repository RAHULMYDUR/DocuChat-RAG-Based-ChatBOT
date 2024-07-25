import streamlit as st
from file_handler import extract_text_from_pdf
from processing import chunk_documents, vectorize_chunks, store_vectors_in_faiss
from retrieval_response import retrieve_relevant_chunks, generate_response

# Define your API key here
api_key = "AIzaSyCzdCOyd-7os-SRgbEolxtwEEgYYkjKpsM"

# Main container for chat history
        st.title("RAG-based Chatbot")

def main():
    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.sidebar.write(f"File uploaded: {uploaded_file.name}")

        # Read the PDF file
        documents = extract_text_from_pdf(uploaded_file)

        # Convert documents to chunks and vectors
        chunks = chunk_documents(documents)
        vectors, vectorizer = vectorize_chunks(chunks)

        # Store vectors in FAISS
        index = store_vectors_in_faiss(vectors)

        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'user_query' not in st.session_state:
            st.session_state.user_query = ""

        
        # Create columns for layout
        chat_col, input_col = st.columns([3, 1])

        with chat_col:
            st.write("## Chat History")
            for chat in st.session_state.chat_history:
                st.write(f"**You:** {chat['question']}")
                st.write(f"**Chatbot:** {chat['answer']}")
                st.write("")

        with input_col:
            st.text_input("Type your question here:", key="user_query", placeholder="Ask a question...")
            st.button("Get Answer", on_click=process_query)

def process_query():
    if st.session_state.user_query:
        # Retrieve data from session state
        user_query = st.session_state.user_query
        
        # Extract and process the data
        retrieved_chunks = retrieve_relevant_chunks(index, chunks, user_query, vectorizer)
        response = generate_response("\n\n".join(retrieved_chunks), user_query, api_key)
        
        # Append the question and answer to the chat history
        st.session_state.chat_history.append({"question": user_query, "answer": response})
        
        # Clear the text input after submission
        st.session_state.user_query = ""

if __name__ == "__main__":
    main()
