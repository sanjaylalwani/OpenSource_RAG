## Advanced RAG techniques used as follow,
1. **Chunk Overlapping**: Chunk overlapping is a technique where a small portion of text from one chunk is included in the next. This helps to maintain contextual continuity between chunks. Check below example,
   
   Original Text:"Large Language Models (LLMs) are powerful AI systems. They are trained on massive datasets to understand and generate human-like text."
  
   Chunking with Overlap (Chunk Size = 20 characters, Overlap = 10 characters)
  
   Chunk 1: "Large Language Models (LLMs) are powerful AI systems."  
   Chunk 2: "are powerful AI systems. They are trained on massive datasets"  
   Chunk 3: "on massive datasets to understand and generate human-like text."

2. 
