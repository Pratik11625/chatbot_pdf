
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
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

# Load environment variables
load_dotenv()

## Load API key and env variable
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")
#input the groq api key
groq_api_key=st.text_input("enter the groq_api_key: ", type="password")

# Initialize the model
llm = ChatGroq(model_name="Gemma-7b-It", groq_api_key=groq_api_key)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the provided context only, please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}"""
)


# # Correctly handle file paths using os.path.join
# base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the base directory where the script is located
# resume_dir = os.path.join(base_dir, "langchain", "chatbot", "resume")

# Function to create vector embeddings and store them in session state
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        # Initialize embeddings, document loader, and vectors in session state
        st.session_state.embeddings = OllamaEmbeddings()
        # st.session_state.loader = PyPDFDirectoryLoader("\resume")
        #   # Data ingestion folder name
        if os.path.exists('resume'):
            st.session_state.loader = PyPDFDirectoryLoader("resume")  # Load PDFs from the resume directory
        else:
            st.error(f"Directory not found: {'resume'}")  # Handle the case where the directory doesn't exist

        # Load documents and check if the list is not empty
        st.session_state.doc = st.session_state.loader.load()
        if not st.session_state.doc:
            st.warning("No documents found in the specified directory.")
            return  # Early exit if no documents are found

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.doc[:50])
        
        # Check if documents were split correctly
        if not st.session_state.final_doc:
            st.warning("Document splitting resulted in no chunks.")
            return  # Early exit if document splitting failed

        # Initialize vectors
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_doc, st.session_state.embeddings)
        st.write("Vector embeddings created and stored in session state.")

# Streamlit app title
st.title("RAG DOC Q&A with Groq and Llama")

# User prompt input
user_prompt = st.text_input("Enter your query:")

# Button to create document embeddings
if st.button("Create Doc Embeddings"):
    create_vector_embeddings()

# Check if user input is provided and embeddings are initialized
if user_prompt and "vectors" in st.session_state:
    # Create document chain and retriever chain
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, doc_chain)

    # Measure the response time
    start = time.process_time()
    response = retriever_chain.invoke({'input': user_prompt})

    # Check if the response has the expected structure and contains an answer
    if 'answer' not in response or not response['answer']:
        st.warning("No response found.")
    else:
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")
        st.write(response['answer'])

        # Optionally, expand to show document similarity search results
    with st.expander("Document Similarity Search"):
        
        # for i, doc in enumerate(response['answer']):
        #         st.write(doc.page_content)
        #         st.write("------------------")

        for i, doc in enumerate(response.get('docs', [])):
            # Check if the object has 'page_content' attribute
            if hasattr(doc, 'page_content'):
                st.write(f"Document {i + 1}: {doc.page_content}")
            else:
                st.write(f"Document {i + 1}: {doc}")  # In case it's a string or other object
            st.write("------------------")


# Handle the case where user input is provided but embeddings are not initialized
elif user_prompt:
    st.warning("Please create document embeddings first.")
