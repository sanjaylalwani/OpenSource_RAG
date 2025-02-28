## Advanced RAG techniques used as follow,
1. **Chunk Overlapping**: Chunk overlapping is a technique where a small portion of text from one chunk is included in the next. This helps to maintain contextual continuity between chunks. Check below example,
   
   Original Text:"Large Language Models (LLMs) are powerful AI systems. They are trained on massive datasets to understand and generate human-like text."
  
   Chunking with Overlap (Chunk Size = 20 characters, Overlap = 10 characters)
  
   Chunk 1: "Large Language Models (LLMs) are powerful AI systems."  
   Chunk 2: "are powerful AI systems. They are trained on massive datasets"  
   Chunk 3: "on massive datasets to understand and generate human-like text."

2. **Re-Ranking**: ReRanking techniques, ensure the most relevant chunks are provided to the LLM. This method uses more precise and time-intensive methods to reorder documents effectively, increasing the similarity between the query and the top-ranked documents.
3. **Metadata Filtering**: This is a simple, trusted, and widely used method in the industry. In RAG, in most of the cases knowledge base contains additional information, metadata, such as author, timestamp, different topic labels. We can use these metadata information for filtering. Different filter options could provided to the users, and system can extract chunks from vector database with provided filters. These can help get most relevant, high quality chunks are extracted, and thus helping LLMs to provide more accurate and relevant answer. This helps a lot to set context better.
