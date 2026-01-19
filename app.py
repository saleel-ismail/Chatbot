import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain.embeddings.base import Embeddings

st.set_page_config(page_title="CHATBOT", layout="centered")

st.title("CHATBOT")
st.write("Upload multiple PDFs")

MASTER_PASSWORD = "12345"
API_KEY ="AIzaSyB0aMKYQ7nTVxl29BohGgSTqbYSy5hNYS4"

password = st.text_input("Enter Access Password:", type="password")
if password != MASTER_PASSWORD:
    st.warning("Enter correct password to continue.")
    st.stop()

genai.configure(api_key=API_KEY)

class GeminiEmbedding(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            r = genai.embed_content(
                model="text-embedding-004",
                content=t,
                task_type="retrieval_document"
            )
            embeddings.append(r["embedding"])
        return embeddings

    def embed_query(self, text):
        r = genai.embed_content(
            model="text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return r["embedding"]

embedding_model = GeminiEmbedding()

def load_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def ask_gemini(prompt):
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_pdfs:

    all_text = ""

    with st.spinner("Reading all PDFs..."):
        for pdf in uploaded_pdfs:
            all_text += load_pdf_text(pdf) + "\n"

    st.success("PDF text extracted successfully")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = splitter.split_text(all_text)

    st.success(f"Total Chunks: {len(chunks)}")

    with st.spinner("Building LangChain FAISS Vector Store..."):
        vectorstore = LCFAISS.from_texts(chunks, embedding_model)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("system ready")

    st.subheader("Tools")

    tool = st.selectbox(
        "Choose a tool",
        [
            "Ask Question",
            "Summarizer",
            "Keyword Extractor",
            "Translate Text",
            "Explain"
        ]
    )

    if tool == "Ask Question":
        user_q = st.text_input("Ask anything to the PDF")

        if st.button("Ask") and user_q.strip():

            retrieved_docs = retriever.invoke(user_q)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            chat_memory = "\n".join(
                [f"USER: {x['q']}\nAI: {x['a']}" for x in st.session_state["chat_history"]]
            )

            prompt = f"""
You are an assistant helping the user understand uploaded PDFs.

RULES:
- Answer only from context.
- If answer not in context say:"thease content is not in pdf" and give own answer and say: "content from outside"
- Keep explanation simple.
- 

CHAT HISTORY:
{chat_memory}

CONTEXT:
{context}

QUESTION:
{user_q}

ANSWER:
"""

            answer = ask_gemini(prompt)

            st.session_state["chat_history"].append({"q": user_q, "a": answer})

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Retrieved Context"):
                st.write(context)

    if tool == "Summarizer":
        if st.button("Summarize All PDFs"):
            prompt = f"""
Summarize this PDF content simply:

TEXT:
{all_text}

SUMMARY:
"""
            summary = ask_gemini(prompt)
            st.write(summary)

    if tool == "Keyword Extractor":
        if st.button("Extract Keywords"):
            prompt = f"""
Extract top 10 important keywords:

{all_text}

FORMAT:
- keyword
- keyword
- keyword
"""
            keywords = ask_gemini(prompt)
            st.write(keywords)

    if tool == "Translate Text":
        target_lang = st.text_input("Enter language (Hindi, Kannada, Tamil, Malayalam, etc.)")

        if st.button("Translate"):
            prompt = f"""
Translate the introduction content to {target_lang}:

{all_text}

TRANSLATION:
"""
            translated = ask_gemini(prompt)
            st.write(translated)

    if tool == "Explain":
        topic = st.text_input("Enter topic to explain")

        if st.button("Explain"):
            prompt = f"""
Explain this topic in simple language with example.

TOPIC: {topic}

CONTEXT:
{all_text}

EXPLANATION:
"""
            explanation = ask_gemini(prompt)
            st.write(explanation)

    if st.button("Clear Chat Memory"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared!")

    with st.expander("Chat History"):
        st.write(st.session_state["chat_history"])
