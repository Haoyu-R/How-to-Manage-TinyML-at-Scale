from SPARQLWrapper import SPARQLWrapper, JSON
from sparql_queries import *
import json

# Put the repository URL from Graph DB here
sparql = SPARQLWrapper(r"http://192.168.56.1:7200/repositories/tinyml_2022")

# Put the query here
sparql.setQuery(query_3)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# Print out the result
for result in results["results"]["bindings"]:
    print(json.dumps(result, indent=4))