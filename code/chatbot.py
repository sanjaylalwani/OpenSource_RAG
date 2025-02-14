import os
import json
import logging
import streamlit as st
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"),)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

st.title("India Budget 25 Bot")
# Initialize chat history
if "messages" not in st.session_state:
    with st.chat_message("user"):
        st.write("Hello 👋")
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
input = st.chat_input("Say something")
if input:
    hits = qdrant_client.search(collection_name="rag_budget_25",
        query_vector=encoder.encode(input),
        limit=3,)

    results = [point.payload["content"] for point in hits]
    context_text = "\n".join(results)

    updated_prompt = f"Answer the following question concisely. Answer should be based on the context:\n\nContext:\n{context_text}\n\nQuestion: {input}\nAnswer:"
 
    # Initialize Groq LLM
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

    # Create a simple prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are chatbot which help to understand the budget presented by Indian Finance Minister for year 2025-2026."),
        ("human","{newprompt}")])

    # Create the chain that guarantees JSON output
    chain = prompt | llm 
    def parse_product(description: str) -> dict:
        result = chain.invoke({"newprompt": description})
        return result
            
    output = parse_product(updated_prompt)

    # Display user message in chat message container
    st.chat_message("user").markdown(input)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": input})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(output.content)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output.content})
