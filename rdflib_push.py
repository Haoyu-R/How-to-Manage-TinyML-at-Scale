import rdflib
import requests

g = rdflib.Graph()
# Specific which ttl file is to be pushed to the Graph DB
g.parse(r"tfliteModels_new.ttl")
# print(g.serialize(format='turtle').decode())

# Put the repository URL from Graph DB here
repo_url = "http://192.168.35.1:7200/repositories/NN_usecase"
url = repo_url + "/statements"

# Use a request to post the ttl to Graph DB
headers = {'Content-type': 'text/turtle',}
response = requests.post(url, headers=headers, data=g.serialize(format='turtle'))
g.close()