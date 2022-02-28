from rdflib import Graph

# Convert a JSONLD-TD into RDF format

td_json_ld = """
{
  "@context": [
    "https://www.w3.org/2019/wot/td/v1",
    {
      "ssn": "http://www.w3.org/ns/ssn/",
      "ssn-system": "http://www.w3.org/ns/ssn/systems",
      "s3n": "http://w3id.org/s3n/",
      "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
      "schema": "https://schema.org/",
	  "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
      "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
      "demo": "https://w3id.org/tinyml-schema-collab/semanticExample/festo_demo#",
	  "nnet": "https://w3id.org/tinyml-schema-collab/neural-network-schema#",
	  "cep": "https://w3id.org/tinyml-schema-collab/cep-rule-schema#" ,
      "@language": "en"
    }
  ],
  "@type": [
    "s3n:SmartSensor"
  ],
  "title": "SSI_Web_198_023_042_021",
  "description": "SIEMENS SSI-Web Node",
  "ssn:hasSubSystem": [
    {
      "@type": "s3n:MicroController",
      "rdfs:comment": "MicroController in Siemens SSI-Web Node.",
      "s3n:hasSystemCapability": [
        {
          "ssn-system:hasSystemProperty": {
            "@type": "s3n:Memory",
            "schema:value": 54,
            "schema:unitCode": "om:kilobyte"
          }
        }
      ],
      "ssn:implements": [
        {
          "@type": "s3n:Algorithm",
		  "@type": "nnet:NeuralNetwork",
          "rdfs:comment": "Neural network for detecting anomaly on a conveyor belt.",
          "s3n:hasProcedureFeature": {
            "ssn-system:inCondition": {
              "@type": "s3n:Memory",
              "rdfs:comment": "RAM requirement.",
              "schema:minValue": "20",
              "schema:unitCode": "om:kilobyte"
            }
          }
        }
      ]
    },
    {
      "@type": "demo:VibrationSensor"
    },
    {
      "@type": "demo:TemperatureSensor"
    }
  ],
  "ssn:isHostedBy": "demo:FESTO_Workstation",
  "securityDefinitions": {
    "nosec_sc": {
      "scheme": "nosec"
    }
  },
  "security": "nosec_sc",
  "properties": {
    "ConveyorBeltState": {
      "title": "ConveyorBeltState",
      "observable": true,
      "readOnly": true,
      "description": "State of the conveyor belt on FESTO workstation based on a vibration sensor.",
      "type": "integer",
	  "@type": "nnet:NetworkOutput",
      "forms": [
        {
          "op": [
            "readproperty",
            "observeproperty"
          ],
          "href": "http://example.org/SSI_Web_198_023_042_021/conveyor-belt-state"
        }
      ]
    },
    "Vibration": {
      "title": "Vibration",
      "observable": true,
      "readOnly": true,
      "description": "Vibration data.",
      "type": "number",
      "maximum": 4,
	  "minimum": -4,
      "unit": "om:gravity",
      "forms": [
        {
          "op": [
            "readproperty",
            "observeproperty"
          ],
          "href": "http://example.org/SSI_Web_198_023_042_021/vibration"
        }
      ]
    },
    "Temperature": {
      "title": "Temperature",
      "observable": true,
      "readOnly": true,
      "description": "Temperature data.",
      "type": "number",
      "minimum": -30,
      "maximum": 60,
      "unit": "om:degree_Celsius",
      "forms": [
        {
          "op": [
            "readproperty",
            "observeproperty"
          ],
          "href": "http://example.org/SSI_Web_198_023_042_021/temperature"
        }
      ]
    }
  },
  "actions": {},
  "events": {},
  "id": "https://w3id.org/tinyml-schema-collab/semanticExample/SSI_Web_198_023_042_021",
  "forms": []
}
"""

g = Graph().parse(data=td_json_ld, format='json-ld')
print(g.serialize(format='turtle'))


