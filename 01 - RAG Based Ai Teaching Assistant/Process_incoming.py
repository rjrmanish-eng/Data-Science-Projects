import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import requests

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:latest"


# ---------------- LOAD DATA ----------------
try:
    df = joblib.load("embeddings.joblib")
    print(f"✅ Loaded {len(df)} records")
except:
    print("⚠️ embeddings file not found, rebuilding...")
    df = joblib.load("data.joblib")  # 👉 apna raw data file


# ---------------- CREATE EMBEDDINGS ----------------
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text_list
        }
    )

    data = r.json()

    if "embeddings" in data:
        return data["embeddings"]
    else:
        print("❌ Embedding error:", data)
        return None


# ---------------- BUILD EMBEDDINGS ----------------
if "embedding" not in df.columns:
    print("🔄 Building embeddings...")
    embeddings = create_embedding(df["text"].tolist())

    if embeddings is None:
        print("❌ Failed. Check model install.")
        exit()

    df["embedding"] = embeddings
    joblib.dump(df, "embeddings.joblib")
    print("✅ Embeddings saved!")


# ---------------- LLM ----------------
def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return r.json()["response"]


# ---------------- TIME FORMAT ----------------
def sec_to_time(sec):
    return f"{int(sec)//60:02d}:{int(sec)%60:02d}"


# ---------------- USER QUERY ----------------
query = input("\n❓ Ask a Question: ")

query_embedding = create_embedding([query])[0]


# ---------------- SIMILARITY ----------------
similarities = cosine_similarity(
    np.vstack(df["embedding"]),
    [query_embedding]
).flatten()

top_idx = similarities.argsort()[::-1][:5]
results = df.iloc[top_idx].copy()

results["time"] = results["start"].apply(sec_to_time)


# ---------------- PROMPT ----------------
prompt = f"""
Answer ONLY from given data:

{results[['title','number','time','text']].to_json(orient='records')}

Question: {query}

Format:
Video: <number> - <title>
Time: <mm:ss>
Explanation: <answer>

If not found:
This topic is not covered in the course.
"""


# ---------------- RESPONSE ----------------
print("\n🤖 Generating answer...")
response = inference(prompt)

print("\n===== ANSWER =====\n")
print(response)