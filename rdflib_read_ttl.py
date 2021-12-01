import rdflib
import os

######## 1. If the ttl file is serialized by rdflib ########
# g = rdflib.Graph()
# g.parse(os.path.join(os.path.dirname(os.path.abspath(__file__)) , "semantic_schema", r"tflite_models.ttl"))
# print(g.serialize(format='turtle').decode())

######## 2. If the ttl file is just in a plain text format ########
with open(os.path.join(os.getcwd(), "semantic_schema", "tflite_models.ttl"), 'r') as f:
    print(f.read())