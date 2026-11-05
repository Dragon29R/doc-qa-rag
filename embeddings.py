import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()  # Load environment variables from .env file

# Initialize the Gemini client
client = genai.Client()
EMBEDDING_MODEL = "gemini-embedding-2"


def get_embedding(text: str) -> np.ndarray:
    """Generate the embedding for a given text and return it as a numpy array."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    # The API returns values in the values property of the first embedding
    return np.array(response.embeddings[0].values)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return float(dot_product / (norm_v1 * norm_v2))


# 1. Define 8 short sentences across 3 distinct topics
# Topics: Astronomy/Space (0, 1, 2), Cooking/Culinary (3, 4, 5), Software/Programming (6, 7)
documents = [
    "Jupiter is the largest planet in our solar system.",        # 0: Space
    "Astronomers discovered a new distant galaxy.",               # 1: Space
    "The James Webb telescope captured images of stars.",         # 2: Space
    "A traditional bread recipe requires flour, water, and salt.",# 3: Cooking
    "The chef prepared a dish of handmade Italian pasta.",       # 4: Cooking
    "Chocolate cake with fudge frosting is the perfect dessert.", # 5: Cooking
    "Python is a popular programming language for AI.",           # 6: Programming
    "The software engineer fixed the bug before deploying.",      # 7: Programming
]

print("Generating embeddings for the 8 documents...")
embeddings = [get_embedding(doc) for doc in documents]
embeddings_matrix = np.array(embeddings)  # shape: (8, embedding_dim)


# 2. Cosine Similarity Matrix between all document pairs
# Similarity matrix = (A . A.T) / (norms)
norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
normalized_embeddings = embeddings_matrix / norms
similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

print("\n--- Cosine Similarity Matrix (8x8) ---")
print("Legend: [0-2: Space] | [3-5: Cooking] | [6-7: Programming]\n")

header = "      " + " ".join([f"[{i}]  " for i in range(8)])
print(header)
for i, row in enumerate(similarity_matrix):
    row_str = " ".join([f"{val:.2f}" for val in row])
    print(f"[{i}]  {row_str}")


# 3. Semantic Retrieval Function
def search(query: str, top_k: int = 3):
    print(f"\n================ Search Query: '{query}' ================")
    q_emb = get_embedding(query)
    q_norm = q_emb / np.linalg.norm(q_emb)

    # Calculate cosine similarity between query and all stored documents
    scores = np.dot(normalized_embeddings, q_norm)

    # Sort descending by score
    ranked_indices = np.argsort(scores)[::-1]

    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        print(f"{rank}. [Score: {scores[idx]:.4f}] (Doc {idx}) {documents[idx]}")


# 4. Standard Case: Direct query
search("What instruments are used to observe the cosmos?")

# 5. Challenging Case 1: Same semantic meaning without shared keywords
search("How to bake dough in the oven?")

# 6. Challenging Case 2: Negation (classic semantic retrieval pitfall)
search("Show me something that is NOT about astronomy or planets.")