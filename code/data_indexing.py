import os
import json
import logging
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

file_path = os.path.join("..", "data", "budget_speech_25.pdf")
reader = PdfReader(file_path)
number_of_pages = len(reader.pages)

# Load environment variables from .env file
load_dotenv()

 
print(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), 
                            api_key=os.getenv("QDRANT_API_KEY"),)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

qdrant_client.create_collection(
    collection_name='rag_budget_25',
    vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(),  
        distance=models.Distance.COSINE)
)

text = ""
for i in range(4, number_of_pages):
    page = reader.pages[i]
    text += page.extract_text()

budget_list = list()
cnt = 0
logging.info("---Started getting data from source-----")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
chunks = text_splitter.split_text(text)

for i, chunk in enumerate(chunks):
    budget_list.append({'page':str(i+3),
                            'content':chunk                        
                            })

# for i in range(4, number_of_pages):
#     page = reader.pages[i]
#     text = page.extract_text()

idx = 0
for doc in budget_list:
    idx = idx+1
    
    qdrant_client.upsert(
        collection_name="rag_budget_25",
        points=[
            models.PointStruct(
                id=idx,
                payload=doc,
                vector=encoder.encode(doc['content']).tolist(),
            ),
        ],
    )
logging.info("---Completed Insert data into the collection-----")
