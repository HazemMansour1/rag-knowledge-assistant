from pathlib import Path

document = Path("data/company_faq.txt").read_text(
    encoding="utf-8"
)


chunks = [chunk.strip()
          for chunk in document.split("\n\n")
          if chunk.strip()]



from sentence_transformers import SentenceTransformer
# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert all chunks into vectors
embeddings = model.encode(chunks)




import numpy as np

def retrieve(query, top_k=2):

    # Convert the user's question into an embedding
    query_embedding = model.encode(query)

    # Calculate cosine similarity against all chunks
    similarities = (
        embeddings @ query_embedding
        / (
            np.linalg.norm(embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )
    )

    # Find indices of the most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return the matching chunks and scores
    results = [
        (chunks[i], float(similarities[i]))
        for i in top_indices
    ]

    return results



import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


def generate_answer(question):

    # Step 1: Retrieve relevant chunks
    results = retrieve(question, top_k=2)

    # Step 2: Combine retrieved chunks into context
    context = "\n\n".join(
        chunk for chunk, score in results
    )

    # Step 3: Send context and question to the LLM
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a company knowledge assistant. "
                    "Answer questions using ONLY the provided context. "
                    "If the context does not contain the answer, "
                    "say 'I don't know based on the provided documents'. "
                    "Do not invent information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


question = "How many vacation days do employees get?"

answer = generate_answer(question)

print("\nRAG Answer:", answer)



if __name__ == "__main__":

    print("Welcome to NovaTech RAG Assistant!")

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer = generate_answer(question)

        print("\nAnswer:", answer)

        print("\nRetrieved Sources:")
        for chunk, score in retrieve(question):
            print(f"- [{score:.3f}] {chunk}")
