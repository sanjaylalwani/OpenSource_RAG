import os
import json
import logging
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

folder_path = os.path.join("..", "data", "target")
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Load environment variables from .env file
load_dotenv()

print(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"),)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

col_name = 'rag_budget'
if qdrant_client.collection_exists(collection_name=col_name):
    print("---Got collection from db-----")
    qdrant_client.get_collection(col_name)
else:
    qdrant_client.create_collection(collection_name=col_name,
    vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(),  
        distance=models.Distance.COSINE))


for fn in file_names:
    file_path = os.path.join(folder_path, fn)
    print(file_path)
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    text = data["content"]
    budget_list = list()
    cnt = 0
    print("---Started getting data from source-----")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        budget_list.append({'page':str(i+3), 'content':chunk })
    idx = 0
    for doc in budget_list:
        idx = idx+1
        
        qdrant_client.upsert(
            collection_name=col_name,
            points=[
                models.PointStruct(
                    id=idx,
                    # payload=doc,
                    vector=encoder.encode(doc['content']).tolist(),
                    payload={"year": data["metadata"]["year"], "type": data["metadata"]["type"]},
                ),
            ],
        )
    logging.info("---Completed Insert data into the collection-----")