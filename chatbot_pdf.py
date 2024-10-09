# Import necessary libraries
!pip install --upgrade langchain

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Updated imports for Vectorstores
from langchain.vectorstores import FAISS  # Chroma and FAISS are under `langchain.vectorstores`

# Import Chat History and Prompts
from langchain.memory import ChatMessageHistory  # Replaces `langchain_community.chat_message_histories`
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import for Groq and HuggingFace LLMs
from langchain.groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings  # Updated import for embeddings

# For loading PDF documents and text splitting
from langchain.document_loaders import PyPDFLoader  # Document loader is now under `langchain.document_loaders`
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Updated from `langchain_text_splitters`

# For managing environment variables
import os
from dotenv import load_dotenv


# Load environment variables from a .env file
load_dotenv()

# Set up HuggingFace embeddings model using an environment variable for the API token
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use HuggingFace's MiniLM model for embeddings

# Set up Streamlit web app interface
st.title("Conversational RAG with PDF upload and chat history")
st.write("Upload PDFs and chat with their content")

# Get the Groq API key from the user input (hidden as a password field)
api_key = st.text_input("Enter the Groq API key: ", type="password")

# If API key is provided, initialize the language model (LLM)
if api_key:
    llm = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=api_key)

    # Chat interface to get session ID from the user
    session_id = st.text_input("Session ID", value="default_session")

    # Manage chat history using Streamlit's session state
    if "store" not in st.session_state:
        st.session_state.store = {}

    # File uploader to allow users to upload PDFs
    upload_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    # Process the uploaded file(s)
    if upload_file:
        document = []
        for upload in upload_file:
            temp_pdf = f"./temporary.pdf"  # Save the uploaded file temporarily
            with open(temp_pdf, "wb") as file:
                file.write(upload.getvalue())
                file_name = upload.name

            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            document.extend(docs)

            st.success("PDF file processed successfully!")
            st.write(f"Processed {len(docs)} documents.")

        # Split the PDF into smaller chunks for embedding
        text_split = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        split = text_split.split_documents(document)

        # Create vectorstore from documents using FAISS for retrieval
        vectorstore = FAISS.from_documents(documents=split, embedding=embedding)
        retriever = vectorstore.as_retriever()

        # Define system prompt for contextualizing the user’s question based on chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question that can be understood without the chat history. "
            "Do not answer the question, just reference it if needed; otherwise return 'I don't know'."
        )

        # Define prompt for contextualizing the question
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create a retriever aware of chat history
        history_aware_retrieve = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Define system prompt for answering questions
        system_prompt = (
            "You are an assistant for question-answer tasks. "
            "Use the following retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )

        # Define prompt for answering questions
        question_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create a chain to answer questions using retrieved documents
        question_answer_chain = create_stuff_documents_chain(llm, question_prompt)
        rag_chain = create_retrieval_chain(history_aware_retrieve, question_answer_chain)

        # Define function to retrieve session history from Streamlit session state
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()  # Initialize chat history if missing
            return st.session_state.store[session_id]

        # Create a runnable with message history for conversational RAG (Retrieval-Augmented Generation)
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",           # Key for user input
            history_messages_key="chat_history",  # Key for chat history
            output_messages_key="answer"          # Expected output key
        )

        # Get user input (question)
        user_input = st.text_input("Your question")
        if user_input:
            # Get session history for the current session ID
            session_history = get_session_history(session_id)

            # Invoke the conversational RAG chain to get the answer
            response = conversational_rag_chain.invoke(
                {"input": user_input},  # Ensure the key "input" matches input_message_key
                config={
                    "configurable": {"session_id": session_id}
                }
            )

            # Display the assistant's answer
            st.success(f"Assistant: {response['answer']}")

else:
    st.warning("Please enter the Groq API key")
