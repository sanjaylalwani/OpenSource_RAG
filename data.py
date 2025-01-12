import os
import json
import logging
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ai_list = ['cs.AI','cs.LG','cs.CV','cs.CL']
print(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), 
                            api_key=os.getenv("QDRANT_API_KEY"),)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

qdrant_client.create_collection(
    collection_name='rag_osc',
    vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(),  
        distance=models.Distance.COSINE)
)

aipapers = list()
cnt = 0
logging.info("---Started getting data from source-----")
with open('arxiv-metadata.json', 'r') as metadata:
    for paper in metadata:
        p = json.loads(paper)
        if p['categories'] in ai_list:
            print(p['categories']) 
            cnt+=1           
            aipapers.append({'authors':p['authors'],
                            'title':p['title'],
                            'categories':p['categories'],
                            'abstract':p['abstract'],
                            'update_date':p['update_date']                         
                            })
print(cnt) 



# qdrant_client.upsert(
#     collection_name='rag_osc',
#     points=[
#         models.PointStruct(
#             id=i, 
#             vector=emb.tolist(), 
#             payload=aipapers
#         )
#         for i, (aipaper, emb) in enumerate(zip(aipapers, embeddings))
#     ]
# )

# # Insert data into the collection
# logging.info("---Started Insert data into the collection-----")
# qdrant_client.upload_points(
#     collection_name="rag_osc",
#     points=[
#         models.PointStruct(
#             id=idx, vector=encoder.encode(doc["abstract"]).tolist(), payload=doc
#         )
#         for idx, doc in enumerate(aipapers)
#     ],
# )

# aipapers = list()
idx = 0
for doc in aipapers:
    idx = idx+1
    
    qdrant_client.upsert(
        collection_name="rag_osc",
        points=[
            models.PointStruct(
                id=idx,
                payload=doc,
                vector=encoder.encode(doc["abstract"]).tolist(),
            ),
        ],
    )
logging.info("---Completed Insert data into the collection-----")