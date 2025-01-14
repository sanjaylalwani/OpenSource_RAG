import os
import json
import logging
from groq import Groq
import streamlit as st
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), 
                            api_key=os.getenv("QDRANT_API_KEY"),)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

st.title("AI Research Bot")
# Initialize chat history
if "messages" not in st.session_state:
    with st.chat_message("user"):
        st.write("Hello 👋")
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
input = st.chat_input("Say something")
if input:

    hits = qdrant_client.search(
        collection_name="rag_osc",
        query_vector=encoder.encode(input),
        limit=3,)

    qdrnt_response = ""
    for hit in hits:
        qdrnt_response += "Document Text: " + hit.payload['abstract'] + "  \n"


    updated_prompt = "Please give precise answer to the question "+ input + " this is context: " + qdrnt_response 

    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7)

    # Create a simple prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are chatbot which help to understand the current trends in research from different AI areas like Machine Learning, Computer Vision, NLP."),
        ("human","{bot_input}")
    ])

    # Create the chain that guarantees JSON output
    chain = prompt | llm 
    def parse_product(description: str) -> dict:
        result = chain.invoke({"bot_input": description})
        return result
            
    # Example usage
    description = prompt # """Text from Qdrant vector database."""

    output = parse_product(updated_prompt)

    # Display user message in chat message container
    st.chat_message("user").markdown(input)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": updated_prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(output.content)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output.content})