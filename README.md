
# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application built from scratch using Python.

## Features
- Document loading and paragraph-based chunking
- Local embeddings using Hugging Face Sentence Transformers
- Semantic search using cosine similarity
- Grounded answer generation through OpenRouter
- Interactive command-line interface
- Basic retrieval and generation evaluation

## Tech Stack
- Python
- Sentence Transformers (all-MiniLM-L6-v2)
- NumPy
- OpenRouter API

## RAG Architecture
1. Load documents
2. Split documents into chunks
3. Generate embeddings
4. Retrieve relevant chunks using cosine similarity
5. Send retrieved context to the LLM
6. Generate answers grounded in the context

## Getting Started

Install dependencies:

    pip install -r requirements.txt

Create a .env file:

    OPENROUTER_API_KEY=your_key_here

Run the application:

    python main.py

## Evaluation Results
- Retrieval Hit@1: 5/5 (100%) on a small test set
- Generation: Passed 3 manual test cases
- Successfully abstained when information was unavailable

## Current Limitations
- Small fictional knowledge base
- In-memory embeddings
- No persistent vector database
- No automated generation evaluation
- Limited retrieval and hallucination safeguards

## Future Improvements
- Vector database integration
- Source citations
- Improved chunking and retrieval
- Automated evaluation
- Production monitoring
