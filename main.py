import os
from dotenv import load_dotenv
import fitz
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# Load local environment variables (if running locally)
load_dotenv()

# Safely fetch Groq API key from Streamlit secrets (for cloud deployment) or environment variables (local)
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(page_title="Pure Hybrid RAG Assistant", page_icon="⚡", layout="centered")

st.title("⚡ Pure Hybrid Search RAG App")
st.write("Dynamic hybrid retrieval combining BM25 keyword matching and Chroma semantic search.")

# Cache the embedding model so it loads only once
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with st.spinner("Loading local embedding model..."):
    embeddings = get_embedding_model()

# 1. File Uploader for PDF
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text and building hybrid search indices..."):
        pdf_bytes = uploaded_file.read()
        pages_text = {}
        full_document_text = ""
        langchain_documents = []
        
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            num_pages = len(doc)
            header_anchor = f"[APPLICANT & PROGRAM HEADER]\n{doc[0].get_text()}\n"
            
            for page_num, page in enumerate(doc):
                p_num = page_num + 1
                page_content = f"\n[Page {p_num}]\n" + page.get_text()
                pages_text[p_num] = page_content
                full_document_text += page_content

        if len(full_document_text.strip()) < 20:
            st.error("This PDF appears to be a scanned image with no text layer. Please use a text-based PDF.")
        else:
            st.success(f"Successfully processed {num_pages} pages dynamically!")

    if len(full_document_text.strip()) >= 20:
        with st.spinner("Creating chunks and indexing BM25 + Chroma..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=80,
                length_function=len
            )
            
            for p_num, p_text in pages_text.items():
                splits = text_splitter.split_text(p_text)
                for split in splits:
                    langchain_documents.append(
                        Document(page_content=split, metadata={"page": p_num})
                    )

            # 1. Dense Vector Store (Chroma)
            vectorstore = Chroma.from_documents(
                documents=langchain_documents,
                embedding=embeddings
            )
            vector_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.6}
            )

            # 2. Sparse Keyword Retriever (BM25)
            bm25_retriever = BM25Retriever.from_documents(langchain_documents)
            bm25_retriever.k = 8
            
            st.success(f"Hybrid indices built successfully across {len(langchain_documents)} chunks!")

        st.markdown("---")

        # 2. Query Input
        user_query = st.text_input("Ask a question about your document:", "What is the Application ID and deposit refund policy?")

        if st.button("Generate Answer") and user_query:
            with st.spinner("Executing optimized hybrid retrieval and page expansion..."):
                
                # Query expansion / query tuning logic
                search_query = user_query
                query_lower = user_query.lower()
                
                if "fraud" in query_lower or "withdraw" in query_lower:
                    search_query = user_query + " fraud withdraw fee refund exception policy"
                elif "deposit" in query_lower:
                    search_query = user_query + " deposit amount duration months"
                elif "work hour" in query_lower:
                    search_query = user_query + " work hours visa fortnight semester"

                # Fetch results from retrievers
                bm25_docs = bm25_retriever.invoke(search_query)
                vector_docs = vector_retriever.invoke(search_query)
                
                # Combine and deduplicate documents while preserving order
                seen_contents = set()
                retrieve_docs = []
                for doc in bm25_docs + vector_docs:
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        retrieve_docs.append(doc)
                
                # Dynamic page expansion from matched metadata
                matched_pages = sorted(list(set(doc.metadata.get("page", 1) for doc in retrieve_docs)))
                
                expanded_context_parts = [header_anchor]
                for p in matched_pages:
                    if p in pages_text:
                        expanded_context_parts.append(pages_text[p])
                        
                context = "\n\n".join(expanded_context_parts)

                # Initialize Groq client
                groq_client = ChatGroq(
                    model="openai/gpt-oss-20b",
                    temperature=0.0,
                    groq_api_key=groq_api_key
                )

                prompt = f"""You are a precise document assistant. Use ONLY the provided document context to answer the question accurately. 
CRITICAL RULE: If a specific detail is not explicitly present in the text, state clearly: "I cannot find that information in the document." Never guess or hallucinate.

Retrieved Document Context:
{context}

Question: {user_query}
"""
                response = groq_client.invoke(prompt)

                st.subheader("Groq Response:")
                st.write(response.content)