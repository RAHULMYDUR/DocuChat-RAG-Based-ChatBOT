import streamlit as st
from file_handler import extract_text_from_pdf
from processing import chunk_documents, vectorize_chunks, store_vectors_in_faiss
from retrieval_response import retrieve_relevant_chunks, generate_response

# Define your API key here
api_key = "AIzaSyCzdCOyd-7os-SRgbEolxtwEEgYYkjKpsM"

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

        # HTML and CSS for layout
        st.markdown("""
        <style>
        .container {
            display: flex;
            flex-direction: column;
            height: 90vh;
            padding: 10px;
        }
        .chat-history {
            flex: 1;
            overflow-y: auto;
        }
        .input-area {
            display: flex;
            align-items: center;
            padding: 10px;
            border-top: 1px solid #ccc;
            background: #f9f9f9;
        }
        .input-area input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .input-area button {
            margin-left: 10px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
        }
        </style>
        """, unsafe_allow_html=True)

        # HTML structure
        st.markdown("""
        <div class="container">
            <div class="chat-history">
                <div id="chat-history">
                    <!-- Chat history will be dynamically updated here -->
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="user-query" placeholder="Type your question here..." />
                <button onclick="getAnswer()">Get Answer</button>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # User query handling and chat history update
        if st.session_state.get('user_query'):
            user_query = st.session_state.user_query
            retrieved_chunks = retrieve_relevant_chunks(index, chunks, user_query, vectorizer)
            response = generate_response("\n\n".join(retrieved_chunks), user_query, api_key)

            # Append the question and answer to the chat history
            st.session_state.chat_history.append({"question": user_query, "answer": response})

            # Clear the text input after submission
            st.session_state.user_query = None

        # Display the chat history
        for chat in st.session_state.chat_history:
            st.write(f"**You:** {chat['question']}")
            st.write(f"**Chatbot:** {chat['answer']}")
            st.write("")

        # JavaScript for handling the form submission and updating chat history
        st.markdown("""
        <script>
        function getAnswer() {
            const userQuery = document.getElementById('user-query').value;
            if (userQuery) {
                // Update Streamlit session state
                streamlit.setComponentValue('user_query', userQuery);
                // Trigger Streamlit rerun
                streamlit.rerun();
            }
        }
        </script>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
