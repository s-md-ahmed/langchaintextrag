# Text-Based RAG Assistant

A production-ready, lightweight web application built with Streamlit and LangChain that implements dynamic hybrid retrieval—combining BM25 keyword matching and Chroma MMR semantic search—without hardcoded shortcuts.

Live App: [https://langchaintextrag-xynbae8dfrpgw5azirkrcv.streamlit.app/]  
*(Note: Hosted on a cloud platform instance, so please allow a few seconds for the application to load on initial request.)*

---

## The Engineering Challenge (Why Pure Hybrid Retrieval?)

Standard vector search excels at conceptual similarity but frequently misses precise exact-keyword lookups (such as specific ID numbers, error codes, or technical names). Conversely, pure keyword search (BM25) fails when a query uses synonyms instead of exact terms. 

This project solves that by merging sparse lexical retrieval with dense vector embedding search using reciprocal document combination, coupled with dynamic page-level context expansion to maintain full layout continuity.

---

## System Architecture & Workflow
<img width="4501" height="7341" alt="diagram (7)" src="https://github.com/user-attachments/assets/ebf7d180-b8f4-4ee5-911d-5868289c1aaa" />


The application executes a robust pipeline: ingesting uploaded PDFs via PyMuPDF (`fitz`), chunking text with sliding overlaps, indexing parallel BM25 and Chroma vector stores, filtering deduplicated search results, expanding context by matching source pages, and querying via the Groq inference engine.

---

## Why This Architecture? (Design Decisions & Trade-offs)

- **Why Hybrid Search (BM25 + Chroma MMR) over Vector-Only Search?**
  - **The Problem:** Pure vector search can drift semantically when looking up rigid identifiers, unique codes, or explicit parameter names.
  - **The Solution:** Fusing BM25 sparse keyword indices with Chroma Maximal Marginal Relevance (MMR) dense retrieval ensures both exact-phrase matching and contextual semantic recall are captured.
- **Why Dynamic Page-Level Context Expansion?**
  - **The Problem:** Standard chunk-only retrieval strips away vital surrounding text, causing isolated chunks to miss overarching document structure.
  - **The Solution:** Retrieved chunk metadata maps directly back to full page dictionaries, dynamically reconstructing complete pages for the model's context window.
- **Why Cached Local HuggingFace Embeddings?**
  - **The Problem:** Fetching embeddings via heavy remote API calls introduces network latency and external service dependencies.
  - **The Solution:** Utilizing `all-MiniLM-L6-v2` locally via HuggingFace with Streamlit resource caching (`@st.cache_resource`) guarantees instantaneous and zero-cost local embedding generation.
- **Why Flexible Environment Secret Management?**
  - **The Problem:** Hardcoding API keys risks credential leaks, while relying solely on local `.env` files breaks cloud deployments.
  - **The Solution:** The app gracefully checks Streamlit cloud secrets (`st.secrets`) first, falling back to local environment variables (`os.getenv`) for smooth cross-environment execution.

---

## Core Features

- **Pure Hybrid Retrieval:** Merges sparse BM25 keyword matching with dense Chroma vector embeddings without hardcoded string-matching hacks.
- **Dynamic Page Expansion:** Automatically pulls full source pages based on matched chunk metadata to preserve document context.
- **Optimized Performance:** Caches local HuggingFace embedding models (`all-MiniLM-L6-v2`) to eliminate redundant load times.
- **Robust PDF Ingestion:** Extracts text via PyMuPDF (`fitz`) with built-in validation checks to intercept scanned image-only PDFs.
- **Configurable LLM Integration:** Powered by ChatGroq for ultra-fast, accurate text generation under strict zero-temperature constraints.

---

## Tech Stack

- **Backend & UI:** Python, Streamlit, PyMuPDF (`fitz`), Python-Dotenv
- **AI / Orchestration:** LangChain, LangChain-Text-Splitters, HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- **Retrieval & Storage:** Chroma Vector Store, BM25 Retriever (`langchain-community`)
- **LLM Engine:** Groq SDK (`ChatGroq`)

---

## Project Structure

```text
text_based_rag/
├── app.py                # Main Streamlit application, hybrid indexing, and retrieval logic
├── requirements.txt      # Python package dependencies
└── .env                  # Local environment variables (Groq API key config)
