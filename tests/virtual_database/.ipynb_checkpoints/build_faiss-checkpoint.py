import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATH
# ============================================================

# build_faiss.py và contexts.json cùng nằm trong virtual_database/
CONTEXTS_FILE = "contexts.json"

# FAISS database sẽ nằm trong virtual_database/faiss_db/
DB_DIR = "faiss_db"

INDEX_FILE = f"{DB_DIR}/index.faiss"
TEXTS_FILE = f"{DB_DIR}/contexts.json"


# ============================================================
# LOAD DATA
# ============================================================

with open(CONTEXTS_FILE, "r", encoding="utf-8") as f:
    contexts = json.load(f)

print(f"Loaded {len(contexts)} contexts")


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

os.makedirs(DB_DIR, exist_ok=True)

# Lưu FAISS index
faiss.write_index(index, INDEX_FILE)

# Lưu contexts theo đúng thứ tự với FAISS index
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