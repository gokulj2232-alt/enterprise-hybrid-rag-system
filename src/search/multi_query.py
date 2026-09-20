import sys

sys.path.insert(0, ".")

from src.search.hybrid_search import hybrid_search

multi_query_search = lambda query, top_k=20: sorted(
{
result.get("document_id"): result
for result in (
hybrid_search(query, top_k=top_k)
+ hybrid_search(query + " explanation", top_k=top_k)
+ hybrid_search(query + " definition", top_k=top_k)
+ hybrid_search(query + " concepts and process", top_k=top_k)
)
if result.get("document_id")
}.values(),
key=lambda x: x.get("hybrid_score", 0),
reverse=True
)
