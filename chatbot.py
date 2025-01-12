import os
import json
import logging
from groq import Groq
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)


# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are chatbot which help to understand the current trends in research from different AI areas like Machine Learning, Computer Vision, NLP."),
    ("user", "{input}")
])

# Create the chain that guarantees JSON output
chain = prompt | llm 
def parse_product(description: str) -> dict:
    result = chain.invoke({"input": description})
    print(result)

        
# Example usage
description = """Text from Qdrant vector database."""

parse_product(description)
