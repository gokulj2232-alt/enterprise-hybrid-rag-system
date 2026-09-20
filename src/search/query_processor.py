import re

QUERY_EXPANSIONS = {
"rag":["retrieval augmented generation","document retrieval","context retrieval","language model"],
"retrieval augmented generation":["RAG","document retrieval","context retrieval","knowledge base","language model"],
"vector database":["embeddings","vector search","similarity search","FAISS"],
"semantic search":["embeddings","vector search","similarity search","natural language search"],
"machine learning":["ML","model training","prediction","algorithms"],
"deep learning":["neural networks","deep neural networks","training","representation learning"],
"natural language processing":["NLP","text processing","language models","text analysis"],
"nlp":["natural language processing","text processing","language models","text analysis"],
"embeddings":["vector representation","semantic representation","dense vectors","similarity"],
"chunking":["document chunking","text splitting","document sections","chunks"],
"faiss":["vector index","similarity search","vector database","embeddings"]
}

clean_query = lambda query: re.sub(r"\s+"," ",query.strip())

expand_query = lambda query: {
"original_query":clean_query(query),
"expanded_query":clean_query(query)+" "+" ".join(dict.fromkeys([term for concept,terms in QUERY_EXPANSIONS.items() if concept in clean_query(query).lower() for term in terms if term.lower() not in clean_query(query).lower()])),
"expansion_terms":list(dict.fromkeys([term for concept,terms in QUERY_EXPANSIONS.items() if concept in clean_query(query).lower() for term in terms if term.lower() not in clean_query(query).lower()]))
}

process_query = lambda query: expand_query(query)

test_queries = ["What is RAG?","How does semantic search work?","What are embeddings?","What is the capital of France?"]

results = [process_query(query) for query in test_queries]

print("QUERY PROCESSOR TEST")
print("=" * 60)

print("Original Query :",results[0]["original_query"])
print("Expanded Query :",results[0]["expanded_query"])

print()
print("Original Query :",results[1]["original_query"])
print("Expanded Query :",results[1]["expanded_query"])

print()
print("Original Query :",results[2]["original_query"])
print("Expanded Query :",results[2]["expanded_query"])

print()
print("Original Query :",results[3]["original_query"])
print("Expanded Query :",results[3]["expanded_query"])
print("No expansion terms required.")
