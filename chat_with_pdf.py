# Import necessary libraries  
import streamlit as st
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

# Set up HuggingFace embedding model# Set up HuggingFace embeddings model using an environment variable for the API token
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]  # Use '=' for assignment
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit app
st.title("Conversational RAG with PDF Upload and Chat History")
st.write("Upload PDFs and chat with their content.")

# Input the Groq API key
api_key = st.text_input("Enter the Groq API Key:", type="password")

# Check if Groq API key is provided
if api_key:
    # Initialize Groq LLM model
    llm = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=api_key)

    # Chat interface for session management
    session_id = st.text_input("Session ID", value="default_session")

    # Set up chat history management
    if "store" not in st.session_state:
        st.session_state.store = {}

    upload_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    # Process uploaded PDF files
    if upload_file:
        documents = []
        for uploaded_file in upload_file:
            temp_pdf = f"./temporary.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

            st.success(f"Processed {len(docs)} documents from {uploaded_file.name}!")

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        split_documents = text_splitter.split_documents(documents)

        # Set up FAISS vectorstore for the document retrieval
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embedding)
        retriever = vectorstore.as_retriever()

        # Define prompts for context and question answering
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question that can be understood without the chat history. "
            "Do not answer the question, just rephrase it, or return 'I don't know' if not possible."
        )

        # ChatPromptTemplate for contextualizing questions
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ('human', "{input}"),
            ]
        )

        # History-aware retriever
        history_aware_retrieve = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Keep the answer concise with a maximum of three sentences.\n\n{context}"
        )

        question_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ('human', "{input}"),
            ]
        )

        # Create document chain and retrieval chain
        question_answer_chain = create_stuff_documents_chain(llm, question_prompt)
        rag_chain = create_retrieval_chain(history_aware_retrieve, question_answer_chain)

        # Function to get session chat history
        def get_session_history(session: str) -> ChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()  # Initialize if missing
            return st.session_state.store[session]

        # RAG chain with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",         # Key for user input
            history_messages_key="chat_history",  # Key for chat history
            output_messages_key="answer"         # Expected output key
        )

        # User input for questions
        user_input = st.text_input("Your question")

        if user_input:
            # Get session history
            session_history = get_session_history(session_id)

            # Invoke the RAG chain
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )

            # Display the assistant's answer
            st.success(f"Assistant: {response['answer']}")

else:
    st.warning("Please enter the Groq API key.")
