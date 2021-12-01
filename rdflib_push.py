import rdflib
import requests
import os

# Put the repository URL from Graph DB here
repo_url = "http://192.168.1.108:7200/repositories/tinyml_2022"
file_name = "syntiant_TinyML_board.ttl"
file_path = os.path.join(os.getcwd(), "semantic_schema", file_name)
url = repo_url + "/statements"
headers = {'Content-type': 'text/turtle',}

######## 1. If the ttl file is serialized by rdflib ########
# g = rdflib.Graph()
# # Specific which ttl file is to be pushed to the Graph DB
# g.parse(file_path)
# # print(g.serialize(format='turtle').decode())
# # Use a request to post the ttl to Graph DB
# response = requests.post(url, headers=headers, data=g.serialize(format='turtle'))
# g.close()

######## 2. If the ttl file is just in a plain text format ########
with open(file_path, 'r') as f:
    # Use a request to post the ttl to Graph DB
    response = requests.post(url, headers=headers, data=f.read())
