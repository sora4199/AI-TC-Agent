import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = BASE_DIR.parent / "dataset" / "test.json"

INDEX_FILE = "index.faiss"
TEXTS_FILE = "contexts.json"


# ============================================================
# LOAD DATASET
# ============================================================

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} dataset samples")


# ============================================================
# GET GOLD CHUNKS FROM test.json
# ============================================================

contexts = []
seen = set()

for item in dataset:
    gold_chunk = item.get("gold_chunk")

    if gold_chunk and gold_chunk not in seen:
        contexts.append(gold_chunk)
        seen.add(gold_chunk)

print(f"Loaded {len(contexts)} unique gold chunks")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

embeddings = model.encode(
    contexts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype("float32")

print(f"Embedding shape: {embeddings.shape}")


# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

# Vì embeddings đã normalize:
# Inner Product = Cosine Similarity
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors")


# ============================================================
# SAVE DATABASE
# ============================================================

faiss.write_index(index, str(INDEX_FILE))

with open(TEXTS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        contexts,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\nFAISS database created successfully!")
print(f"Index: {INDEX_FILE}")
print(f"Texts: {TEXTS_FILE}")