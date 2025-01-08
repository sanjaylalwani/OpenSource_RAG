import os
import json
import logging
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer



qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), 
                            api_key=os.getenv("QDRANT_API_KEY"),)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

qdrant_client.create_collection(
    collection_name='rag_osc',
    vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(),  
        distance=models.Distance.COSINE)
)

aipapers = list()
logging.info("---Started getting data from source-----")
with open('arxiv-metadata.json', 'r') as metadata:
    for paper in metadata:
        p = json.loads(paper)
        print(p['authors'])
        aipapers.append({'authors':p['authors'],
                         'title':p['title'],
                         'categories':p['categories'],
                         'abstract':p['abstract'],
                         'update_date':p['update_date']                         
                         })
         
summaries = [aipaper['abstract'] for aipaper in aipapers]
logging.info("---Completed getting data from source-----")
embeddings = encoder.encode(summaries)

# Insert data into the collection
logging.info("---Started Insert data into the collection-----")
qdrant_client.upsert(
    collection_name='rag_osc',
    points=[
        models.PointStruct(
            id=i, 
            vector=emb.tolist(), 
            payload=aipapers
        )
        for i, (aipaper, emb) in enumerate(zip(aipapers, embeddings))
    ]
)

logging.info("---Completed Insert data into the collection-----")
