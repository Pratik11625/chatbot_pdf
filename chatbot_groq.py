import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import time
import tempfile

# Load environment variables
load_dotenv()

# App Title and Sidebar Setup
st.set_page_config(page_title="Chat with PDFs", layout="wide")
st.sidebar.title("Document Chat with Groq and Llama")
st.sidebar.write("Upload your document, create embeddings, and ask questions.")

# API Key Input (from sidebar)
groq_api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

# File Uploader for PDF and Text Files
uploaded_files = st.sidebar.file_uploader("Upload PDF/Text Files", type=["pdf", "txt"], accept_multiple_files=True)

# Initialize the LLM model
if groq_api_key:
    llm = ChatGroq(model_name="Gemma-7b-It", groq_api_key=groq_api_key)
else:
    st.sidebar.warning("Please enter your Groq API key.")
    st.stop()

# Initialize session state for embeddings and chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# Function to create vector embeddings from uploaded documents
def create_vector_embeddings(docs):
    st.session_state.embeddings = OllamaEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    final_docs = text_splitter.split_documents(docs)

    if final_docs:
        st.session_state.vectors = FAISS.from_documents(final_docs, st.session_state.embeddings)
        st.sidebar.success("Embeddings created successfully!")
    else:
        st.sidebar.error("Document splitting failed. Try uploading a valid document.")

# # Process uploaded files
# if uploaded_files:
#     documents = []
#     for file in uploaded_files:
#         if file.type == ".pdf":
#             # Load PDF files
#             loader = PyPDFLoader(file)
#             docs = loader.load()
#             documents.extend(docs)
#         elif file.type == ".txt":
#             # Load text files
#             text = file.read().decode("utf-8")
#             documents.append({"page_content": text, "metadata": {"source": file.name}})
    
#     # Create embeddings when the "Create Embeddings" button is clicked
#     if st.sidebar.button("Create Embeddings"):
#         create_vector_embeddings(documents)
# Process uploaded files (PDF or Text files)
if uploaded_files:
    documents = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            # Save the uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.getvalue())
                temp_pdf_path = temp_pdf.name
            
            # Load the PDF using the saved temporary file path
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            documents.extend(docs)
            
            # Optionally remove the temporary file after processing
            os.remove(temp_pdf_path)

        elif file.type == "text/plain":
            # Load text files
            text = file.read().decode("utf-8")
            documents.append({"page_content": text, "metadata": {"source": file.name}})
    
    # Create embeddings when the "Create Embeddings" button is clicked
    if st.sidebar.button("Create Embeddings"):
        create_vector_embeddings(documents)

# Chat Input
st.title("Chat with Your Document 📄")
user_prompt = st.text_input("Ask a question based on your document:")

# If embeddings are available, allow users to ask questions
if user_prompt and st.session_state.vectors:
    # Define the prompt template for querying the document
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the provided context from the uploaded document.
        <context>
        {context}
        </context>
        Question: {input}"""
    )

    # Create the document chain and retriever chain
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, doc_chain)

    # Measure response time
    start = time.process_time()
    response = retriever_chain.invoke({'input': user_prompt})

    # Store response in chat history
    st.session_state.chat_history.append({"user": user_prompt, "assistant": response.get('answer', 'No answer found.')})

    # Display response and response time
    st.write(f"Response time: {time.process_time() - start:.2f} seconds")
    st.write(f"**Assistant:** {response.get('answer', 'No answer found.')}")

    # Display chat history
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Assistant:** {chat['assistant']}")

    # Display Document Similarity Context
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get('docs', [])):
            if hasattr(doc, 'page_content'):
                st.write(f"Document {i + 1}: {doc.page_content}")
                st.write("------------------")
else:
    if user_prompt:
        st.warning("Please create document embeddings first by uploading a document.")

# Footer Message
st.sidebar.write("© 2024 Conversational Document Chat")
