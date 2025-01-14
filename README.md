# Ask Your PDF - RAG App with Google LLM

This is a web application built with **Streamlit** and **LangChain**, which allows users to ask questions based on the contents of a PDF document. The app uses a **retrieval-augmented generation** (RAG) approach, combining **ChromaDB** for document storage and retrieval, along with **Google's Generative AI** model to answer user queries.

## Features

- **Upload PDF**: Users can upload PDF files (up to 30 pages).
- **Ask Questions**: Users can ask questions based on the content of the uploaded PDF, and the app will provide answers using Google's Generative AI.
- **RAG Approach**: Combines document retrieval from the uploaded PDF with the powerful response generation model from Google to ensure concise and relevant answers.

## Screenshots
![image](https://github.com/user-attachments/assets/667ed0ec-6052-462a-812f-874d1d29249d)

![image](https://github.com/user-attachments/assets/2dec4954-862f-4738-bd37-68a093f4178e)


## Technologies Used

- **Streamlit**: For building the web application.
- **LangChain**: For document processing, retrieval, and embedding generation.
- **ChromaDB**: A vector database used for efficient document retrieval.
- **Google Generative AI**: For answering questions based on the document content.
- **PyPDF2**: For reading and processing PDFs.
- **Python-dotenv**: For loading environment variables.

## Setup Instructions

### 1. Clone the repository:
git clone https://github.com/your-username/ask-your-pdf-rag-app.git
cd ask-your-pdf-rag-app

### 2. Set up a virtual environment:
python -m venv myenv
source myenv/bin/activate  # For macOS/Linux
myenv\Scripts\activate  # For Windows

### 3. Install the required dependencies:
pip install -r requirements.txt
4. Set up environment variables:
Create a .env file in the root directory of the project and add the following line with your Google Generative AI API key:
GOOGLE_API_KEY=your_google_api_key_here

### 5. Run the Streamlit app:
streamlit run app.py
Visit http://localhost:8501 in your browser to use the app.

## How It Works
Upload a PDF: When a user uploads a PDF, the app checks the number of pages. If the file has more than 30 pages, the app will notify the user to upload a smaller file.
Document Loading and Splitting: The PDF content is loaded, split into chunks using LangChain's RecursiveCharacterTextSplitter, and stored in ChromaDB.
Vector Store: The document chunks are embedded using Google Generative AI Embeddings and stored in ChromaDB for efficient retrieval.
Question-Answering: The user can ask questions about the document, and the app will retrieve the most relevant chunks and use Google’s Generative AI to generate a response.

## Troubleshooting
If you face issues related to ChromaDB or document retrieval, ensure that your environment meets the necessary dependencies and that you have an appropriate API key for Google Generative AI.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
