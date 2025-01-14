import os
import sys
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from chromadb.config import Settings  # For ChromaDB configuration
from dotenv import load_dotenv

# Use pysqlite3 for compatibility
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
load_dotenv()

st.title("Ask questions based on your PDF.")

# File upload component
uploaded_file = st.file_uploader("Upload a PDF file (max 30 pages):", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Read the uploaded PDF to check the number of pages
    pdf_reader = PdfReader(temp_file_path)
    num_pages = len(pdf_reader.pages)

    if num_pages > 30:
        st.error(f"The uploaded PDF has {num_pages} pages, which exceeds the 30-page limit. Please upload a smaller PDF.")
    else:
        # Process the valid PDF
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)

        # Use a temporary directory for ChromaDB persistence
        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                client_settings=Settings(persist_directory=temp_dir)
            )

            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

            # Set up the LLM
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

            # System prompt for the model
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            # Chat input for user queries
            query = st.chat_input("Ask something about the uploaded document: ")

            if query:
                # Create chains for retrieval and answering
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # Get the response
                response = rag_chain.invoke({"input": query})

                # Display the response
                st.write(response["answer"])

    # Clean up the temporary file after processing
    os.remove(temp_file_path)
else:
    st.info("Please upload a PDF file to get started.")
