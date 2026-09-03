# import pandas as pd 
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# import numpy as np
# import requests
# from openai import OpenAI
# from config import API_key
 

# client = OpenAI(api_key=API_key)

# df = joblib.load("embeddings.joblib")

# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding


# def infrance(prompt):
#     r = requests.post("http://localhost:11434/api/generate", json={
#         #  "model": "deepseek-r1",
#         "model": "Llama3.2",
#         "prompt": prompt,
#         "Stream": False

#     })
     
#     response = r.json()
#     print(response)

#     return response


# def infrance_openai():
#     response = client.responses.create(
#     model="gpt-5",
#     input=prompt)

#     return response.output.text


# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0] 

# # Find similarities of question_embedding with other embeddings
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding']).shape)
# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# # print(similarities)
# top_results = 5
# max_index = similarities.argsort()[::-1][0:top_results]
# # print(max_index)
# new_df = df.loc[max_index]
# # print(new_df[["title", "number" ,"text"]])


# # for index , item in new_df.iterrows():
# #     print(index, item['title'], item['number'],item['text'],item['start'],item['end'])




# prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

# {new_df[["title", "number" ,"start","end","text"]].to_json(orient="records")} # YE RESULT KO JSON ME CONVERT KARNA HAI
# ---------------------------------
# "{incoming_query}"
# User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
# '''
# with open("prompt.txt", "w") as f:
#     f.write(prompt)

# response = infrance(prompt)["response"]
# print(response)

# with open('response.txt', 'w') as f:
#     f.write(response)



import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import requests
from openai import OpenAI
from config import API_key


# ---------------- OPENAI CLIENT ----------------
client = OpenAI(api_key=API_key)

# ---------------- LOAD EMBEDDINGS ----------------
df = joblib.load("embeddings.joblib")


# ---------------- CREATE EMBEDDINGS (OLLAMA) ----------------
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    return r.json()["embeddings"]


# ---------------- OPENAI INFERENCE ----------------
def inference_openai(prompt):
    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )
    return response.output.text


# ---------------- USER QUERY ----------------
incoming_query = input("Ask a Question: ")

# Create embedding for user query
question_embedding = create_embedding([incoming_query])[0]

# ---------------- COSINE SIMILARITY ----------------
similarities = cosine_similarity(
    np.vstack(df["embedding"]),
    [question_embedding]
).flatten()

top_results = 5
top_indices = similarities.argsort()[::-1][:top_results]
new_df = df.loc[top_indices]

# ---------------- PROMPT ----------------
prompt = f"""
I am teaching web development in my Sigma web development course.
Here are video subtitle chunks containing video title, video number,
start time in seconds, end time in seconds, and the text at that time:

{new_df[["title","number","start","end","text"]].to_json(orient="records")}

---------------------------------
User Question:
"{incoming_query}"

Answer in a human way and tell:
- Which video the topic is taught in
- At what timestamp
- Guide the user to that specific video

If the question is unrelated, politely say that you can only answer
questions related to the course.
"""

# Save prompt
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

# ---------------- OPENAI RESPONSE ----------------
response = inference_openai(prompt)
print(response)

# Save response
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

