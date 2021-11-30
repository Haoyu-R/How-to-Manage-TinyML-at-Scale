from SPARQLWrapper import SPARQLWrapper, JSON
from sparqlQueries import *
import json

# Put the repository URL from Graph DB here
sparql = SPARQLWrapper(r"http://192.168.35.1:7200/repositories/NN_usecase")

# Put the query here
sparql.setQuery(query_1)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# Print out the result
for result in results["results"]["bindings"]:
    print(json.dumps(result, indent=4))