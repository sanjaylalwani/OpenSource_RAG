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

col_name = 'rag_budget'
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"),)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

def rerank_with_llm(query, documents):
    scores = []
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
    for doc in documents:
        prompt = f"Given the query '{query}', how relevant is the following document? Document: '{doc}'"
        # response = llm.predict(prompt)
        messages = [("system","You are a helpful assistant that helps to rerank the document.",),
            ("human", prompt),]
        
        try:
            ai_msg = llm.invoke(messages)
            score = float(ai_msg.content.strip())
            print(ai_msg.content)
        except ValueError:
            score = 0
        scores.append(score)
    reranked_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
    return reranked_docs

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
    hits = qdrant_client.search(collection_name=col_name, 
                               query_vector=encoder.encode(input).tolist(), 
                               limit=5,)

    results = [point.payload["content"] for point in hits]

    rerank_results = rerank_with_llm(input, results)
    context_text = "\n".join(rerank_results)
    print(context_text)

    updated_prompt = f"Answer the following question concisely. Answer should be based on the context:\n\nContext:\n{context_text}\n\nQuestion: {input}\nAnswer:"
    # Initialize Groq LLM
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

    # Create a simple prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are chatbot which help to understand the budget presented by Indian Finance Ministers for different years from 2014 to 2025. You may have either Interim or Full budget, if its Interim budget, do look for Full budget as well. Interim budget is the budget which is presented before election, and then Full budget is presented.
          Ensure you answers are concise, and to the point always.
          Do not hallucinate, give answer from sources only, if source does not have context then say You dont know the answer."""),
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