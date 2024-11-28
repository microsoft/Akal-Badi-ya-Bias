import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_query_sentences(fname, n_examples=100):
    df = pd.read_csv(fname).sort_values("score", ascending=False)
    sentences = df["text"].tolist()
    return sentences[:n_examples]


def get_key_sentences(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


model = SentenceTransformer("sentence-transformers/LaBSE")
embeddings_np = np.load("hi_sampled.npy")
key_sentences = get_key_sentences("hi_sampled.txt")
query_sentences = get_query_sentences("text_with_scores_combined.csv")

results = [[], [], [], [], []]

d = embeddings_np.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)  # Using L2 distance
index.add(embeddings_np)

for line in tqdm(query_sentences):
    query_embedding = model.encode([line], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=5)
    similar_sentences = [key_sentences[i] for i in I[0]]

    for i, sent in enumerate(similar_sentences):
        results[i].append([line, sent])

for i in range(5):
    df = pd.DataFrame(results[i], columns=["query", "similar_sentence"])
    df.to_excel(f"hi_sampled_mined_{i}.xlsx", index=False)
