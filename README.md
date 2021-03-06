# How to Manage TinyML at Scale

The future of machine learning (ML) is tiny and bright. Embedded ML has risen to popularity in an era. Various neural networks are proposed to run on constrained microcontrollers which can consume sensor data everywhere in real-time.  Embedded devices are typically customized towards specific tasks and are subject to heterogeneity and fragmentation. The deployment of TinyML in production has to take hardware constraints under consideration, such as available onboard sensors and memory.  The management of TinyML systems becomes increasing cumbersome considering the diversity and vast amount of ML models and microcontrollers developed every year. In light of these challenges, we present a framework based on semantics to enable easy co-management of neural network models and embedded devices, from discovering possible combinations and benchmarking to deploying TinyML models on microcontrollers.

For more information on the project, please see our paper
[How to Manage TinyML at Scale](https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale)

![Capture1.PNG](/_resources/Capture1.PNG)

## Citation
If our work has been useful for your research and you would like to cite it in an scientific publication, please cite [How to Manage TinyML at Scale](https://arxiv.org/abs/2202.09113) as follows:
```
@misc{ren2022manage,
      title={How to Manage Tiny Machine Learning at Scale: An Industrial Perspective}, 
      author={Haoyu Ren and Darko Anicic and Thomas Runkler},
      year={2022},
      eprint={2202.09113},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Project Structure
* [collected_models](https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale/tree/main/collected_models): the TinyML  (tflite) models that we collected
* [estimate_tensor_arena_size](https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale/tree/main/estimate_tensor_arena_size): the source file to estimate the RAM consumption given a tflite model
* [model_repo](https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale/tree/main/model_repo): a folder to stimulate a model repo for hosting the parsed tflite models
* [semantic_schema](https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale/tree/main/semantic_schema): the RDF schema / information model we proposed for neural network and embedded devices, as well as the supplementary schema
* Models_Information.xlsx: an excel sheet storing the information of collected tflite models for easier parsing
* bin2tflite.py: convert binary tinyml model to tflite format
* find-arena-size: the binary executable to calculate RAM consumption given a tflite model
* rdflib_push.py: push the semantic representation of TinyML system (NN models or devices) to the knowledge graph hosted in GraphDB
* rdflib_read_ttl.py: pretty print a serialized RDF turtle file
* requirements.txt: use `pip install -r requirements.txt` to install required packages
* semantic_querying.py: use SPARQL to query the knowledge graph hosted in GraphDB
* semantic_utils.py
* sparql_queries.py: example  SPARQL queries
* tflite2semantic_parser_xlsx.py: generate semantic representations of the NN models stored in the folder [collected_models](https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale/tree/main/collected_models) against the [proposed semantic schema](## Semantic Schema of Neural Network) combining the information provided in `Models_Information.xlsx`
* tflite2semantic_user_input.py: generate a semantic representation for each NN model against the [proposed semantic schema](#semantic-schema-of-neural-network) by asking the user a few input questions

## Use

Our project is runnable in a Linux environment, as the binary executable is built on a Linux environment. Alternatively, one can use our [google colab](https://colab.research.google.com/drive/1Cnpoqb92yiERrBMLDjCqxB9Yx7-9ufcu?usp=sharing) script to start the development.

Install the project:

```
git clone 'https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale.git'
```

Install the dependency:
```
pip install -r requirement.txt
```

Run  `tflite2semantic_parser_xlsx.py` to see how the collected models in the [model_repo](https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale/tree/main/model_repo) can be parsed into semantic representation against the [proposed semantic schema](#semantic-schema-of-neural-network) combining the information provided in `Models_Information.xlsx` in one go. Please be aware that the order of the models listed in the the folder `collected_models`  and in the information sheet `Models_Information.xlsx` should both be in alphabetic order and match with each other.

Run  `tflite2semantic_user_input.py` to see how each model can be parsed into semantic representation by answering a few questions in the CMD.

To work with the semantic representations of neural networks and embedded devices, we recommend using [GraphDB free](https://graphdb.ontotext.com/). The scripts `rdflib_push.py`, `semantic_querying.py`, `sparql_queries.py` contain the code and example queries for interacting with GraphDB.

## Semantic Schema of Neural Network

![Capture2.PNG](/_resources/Capture2.PNG)

## To do
```
To be added
```

## Related Project
* [Demo_Arduino_Motion](https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale/tree/main/Demo_Arduino_Motion): the Arduino implementation for loading and deploying tflite micro models from a central device to an Arduino Nano 33 BLE Sense board using BLE on the fly.

## Contributing to the project

We welcome contributions. Please contact us by email to get started!
