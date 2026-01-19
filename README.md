# 🚀 PDF Chatbot using Gemini AI (RAG-based)

PDF Chatbot is an intelligent document interaction system built using Streamlit, Google Gemini AI, and LangChain FAISS.  
It allows users to upload multiple PDF files and interact with them through natural language queries, summaries, keyword extraction, translation, and explanations using Retrieval-Augmented Generation (RAG).

---

## ✨ Features

- Password-protected access  
- Upload and process multiple PDF files  
- Ask questions strictly from PDF content (RAG-based)  
- Context-aware answers using FAISS vector search  
- PDF summarization  
- Keyword extraction  
- Translate PDF content to any language  
- Explain topics in simple language with examples  
- Chat memory support  
- View retrieved context for transparency  

---

## 🏗️ Architecture

User → Streamlit Web App → PDF Upload & Text Extraction  
→ Text Chunking → Gemini Embeddings (text-embedding-004)  
→ FAISS Vector Store → Retriever (Top-K Search)  
→ Gemini 2.5 Flash (Answer Generation)

---

## 🛠️ Technologies Used

### Backend / AI
- Python  
- Google Gemini AI  
- Gemini Embedding Model (text-embedding-004)  
- LangChain  
- FAISS Vector Database  

### Frontend
- Streamlit  

### Libraries
- PyPDF2  
- langchain-community  
- langchain-text-splitters  

---

## 📂 Project Structure

pdf-chatbot  
|-- app.py  
|-- requirements.txt  
|-- README.md  

---

## 🔐 Security

- Password-based access control using Streamlit  
- API key configured securely  
- Unauthorized users are blocked automatically  

---

## 🧠 Tools Available in the App

- Ask Question – Ask anything related to uploaded PDFs  
- Summarizer – Generate a simple summary of all PDFs  
- Keyword Extractor – Extract top 10 important keywords  
- Translate Text – Translate PDF content to any language  
- Explain – Explain a topic using PDF context  

---

## 🧪 Testing

- Upload single or multiple PDFs  
- Ask questions related to document content  
- Verify answers using retrieved context  
- Test summarization, translation, and explanation tools  
- Clear and review chat history  

---

## 🎯 Learning Outcomes

- Building RAG-based AI systems  
- Vector database usage with FAISS  
- Gemini AI integration with custom embeddings  
- PDF text extraction and chunking  
- Prompt engineering with context control  
- Streamlit application development  
- Secure AI application design  

---

## 👤 Author

Ismail Saleel D  
Python Developer | AI & Cloud Enthusiast  
MCA – Data Science & Big Data Analytics  
