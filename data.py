import os
import json
import logging
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pypdf import PdfReader

reader = PdfReader("budget_speech.pdf")
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

budget_list = list()
cnt = 0
logging.info("---Started getting data from source-----")
for i in range(4, number_of_pages):
    page = reader.pages[i]
    text = page.extract_text()
    budget_list.append({'page':str(i),
                            'content':text                        
                            })
 
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