import streamlit as st
from file_handler import extract_text_from_pdf
from processing import chunk_documents, vectorize_chunks, store_vectors_in_faiss
from retrieval_response import retrieve_relevant_chunks, generate_response

# Define your API key here
api_key = "AIzaSyCzdCOyd-7os-SRgbEolxtwEEgYYkjKpsM"


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

        # Layout Containers
        st.title("RAG-based Chatbot")

        # Input and Response Containers
        input_container = st.container()
        response_container = st.container()

        # Function to get user input
        def get_text():
            input_text = st.text_input("You: ", "", key="input")
            return input_text

        # Display input container
        with input_container:
            user_input = get_text()
            if st.button("Get Answer"):
                if user_input:
                    retrieved_chunks = retrieve_relevant_chunks(index, chunks, user_input, vectorizer)
                    response = generate_response("\n\n".join(retrieved_chunks), user_input, api_key)

                    # Append the question and answer to the chat history
                    st.session_state.chat_history.append({"question": user_input, "answer": response})
                    st.session_state.user_query = ""

        # Display response container
        with response_container:
            if st.session_state['chat_history']:
                for chat in st.session_state['chat_history']:
                    st.write(f"**You:** {chat['question']}")
                    st.write(f"**Chatbot:** {chat['answer']}")
                    st.write("")

if __name__ == "__main__":
    main()
