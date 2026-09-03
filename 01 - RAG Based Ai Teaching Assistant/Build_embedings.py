import pandas as pd
import joblib
import requests

EMBED_MODEL = "nomic-embed-text"

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv")
print(f"✅ Loaded {len(df)} rows")


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
        print("❌ Error:", data)
        return None


# ---------------- BUILD ----------------
print("🔄 Building embeddings...")
embeddings = create_embedding(df["text"].tolist())

if embeddings is None:
    print("❌ Failed")
    exit()

df["embedding"] = embeddings

# ---------------- SAVE ----------------
joblib.dump(df, "embeddings.joblib")
print("💾 Saved as embeddings.joblib")