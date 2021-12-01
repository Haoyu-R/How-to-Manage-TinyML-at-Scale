# Make TinyML Manageble

The future of machine learning (ML) is tiny and bright. Embedded ML has risen to popularity in an era. ML models, especially neural networks, are proposed to run on constrained embedded devices, and they can consume sensor data everywhere in real-time.  However, embedded devices are typically customized towards specific tasks and are subject to diversity and fragmentation. In addition, the deployment of TinyML models has to take hardware constraints under consideration, such as available sensors, memory, which makes the management of mass TinyML systems cumbersome considering the vast amount of embedded devices shipped every year. In light of these challenges, we present a framework based on semantics to enable easy co-management of neural network models and embedded devices, from discovering possible combinations and benchmarking to deploying TinyML models on microcontrollers.

For more information on the project, please see our paper
[Make TinyML Manageble]()

![Capture1.png](/_resources/Capture1.png)

## Documentation
* [Demo_Arduino_Motion](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/Demo_Arduino_Motion): the Arduino implementation of loading and deploying TinyML models using BLE on the fly.
* [collected_models](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/collected_models): the TinyML  (tflite) models that we collected
* [estimate_tensor_arena_size](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/estimate_tensor_arena_size): the source file to estimate the RAM consumption given a tflite model
* [model_repo](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/model_repo): a folder to stimulate a model repo for hosting the parsed tflite models
* [semantic_schema](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/semantic_schema): the RDF schema / information model we proposed for neural network and embedded devices, as well as the supplementary schema
* Models_Information.xlsx: an excel sheet storing the information of collected tflite models for easier parsing
* bin2tflite.py: convert binary tinyml model to tflite format
* find-arena-size: the binary executable to calculate RAM consumption given a tflite model
* rdflib_push.py: push the semantic representation of TinyML system (NN models or devices) to the knowledge graph hosted in GraphDB
* rdflib_read_ttl.py: pretty print a serialized RDF turtle file
* requirements.txt: use `pip install -r requirements.txt` to install required packages
* semantic_querying.py: use SPARQL to query the knowledge graph hosted in GraphDB
* semantic_utils.py
* sparql_queries.py: example  SPARQL queries
* tflite2semantic_parser_xlsx.py: generate semantic representations of the NN models stored in the folder [collected_models](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/collected_models) against the [proposed semantic schema](## Semantic Schema of Neural Network) combining the information provided in `Models_Information.xlsx`
* tflite2semantic_user_input.py: generate a semantic representation for each NN model against the [proposed semantic schema](## Semantic Schema of Neural Network) by asking the user a few input questions

## Use

Our project is runnable in a Linux environment, as the binary executable is built on a Linux environment. Alternatively, one can use our [google colab](https://colab.research.google.com/drive/1Cnpoqb92yiERrBMLDjCqxB9Yx7-9ufcu?usp=sharing) script to start the development.

Install the project:

```
git clone 'to be added'
```

Install the dependency:
```
pip install -r requirement.txt
```

Run  `tflite2semantic_parser_xlsx.py` to see how the stored models can be parsed into semantic representation against the information stored in the folder `Models_Information.xlsx` in one go. Please be aware that the order of the models listed in the the folder `collected_models`  and in the information sheet `Models_Information.xlsx` should both be in alphabetic order and match with each other.

Run  `tflite2semantic_user_input.py` to see how each stored model can be parsed into semantic representation by answering a few questions in the CMD.

To store the semantic representations of neural networks and embedded devices, we recommend using [GraphDB free](https://graphdb.ontotext.com/). The scripts `rdflib_push.py`, `semantic_querying.py`, `sparql_queries.py` contain the code and example queries for interacting with GraphDB.

To deploy the tflite model from a central device to an Arduino Nano 33 BLE Sense board using BLE and thus change the model on the board without reflashing the code, please see the instructions in  [Demo_Arduino_Motion]()

## Semantic Schema of Neural Network

![Capture2.png](/_resources/Capture2.png)

## Citation
If our work has been useful for your research and you would like to cite it in an scientific publication, please cite [Make TinyML manageble]() as follows:
```
To be added
```

## Contributing to the project

We welcome contributions. Please contact us by email to get started!





	