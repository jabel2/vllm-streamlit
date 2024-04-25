from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from vllm_interface import llm

graph = Neo4jGraph(
    url="bolt://neo4j:7687", username="neo4j", password="neo4juser"
)

#print(graph.schema)

chain = GraphCypherQAChain.from_llm(
    llm, 
    graph=graph, 
    verbose=True,
    #top_k = 5,
    return_intermediate_steps=True,
    validate_cypher=True
)

#chain.invoke("what businesses have been reviewed?")

# result = chain.invoke("Sort the businesses into types.  What types of businesses are in the graph?")
# print(f"Intermediate steps: {result['intermediate_steps']}")
# print(f"Final answer: {result['result']}")


result = chain.invoke("What distinct cities are the businesses located?")
print(f"Intermediate steps: {result['intermediate_steps']}")
print(f"Final answer: {result['result']}")

# result = chain.invoke("What business names are similar to or have brewery in them?")
# print(f"Intermediate steps: {result['intermediate_steps']}")
# print(f"Final answer: {result['result']}")