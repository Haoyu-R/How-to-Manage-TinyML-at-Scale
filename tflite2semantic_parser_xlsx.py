import glob
import os
import shutil
import uuid
import time
import numpy as np
import tflite
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, SDO
from semantic_utils import *
import tensorflow as tf
import pandas as pd
import os

# Schema reference: https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/lite/schema/schema.fbs/#L344
# Tensor arena size reference: https://github.com/edgeimpulse/tflite-find-arena-size
cmd = "./find-arena-size Hello_World.tflite"

# Create namespace for none-supported namespace by rdflib
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
nnet = Namespace("http://tinyml-schema.org/networkschema#")
s3n = Namespace("http://w3id.org/s3n/")
ssn_system = Namespace("http://www.w3.org/ns/ssn/systems/")
ssn_extend = Namespace("http://tinyml-schema.org/ssn_extend/")
sosa_extend = Namespace("http://tinyml-schema.org/sosa_extend/")
s3n_extend = Namespace("http://tinyml-schema.org/s3n_extend/")
om = Namespace("http://www.ontology-of-units-of-measure.org/resource/om-2/")

# Create a Graph
g = Graph()

# Name binding
nm = g.namespace_manager
nm.bind("sosa", SOSA)
nm.bind("nnet", nnet)
nm.bind("schema", SDO)
nm.bind("s3n", s3n)
nm.bind("ssn", SSN)
nm.bind("om", om)
nm.bind("ssn_extend", ssn_extend)
nm.bind("sosa_extend", sosa_extend)
nm.bind("s3n_extend", s3n_extend)


def tflite2semantic(path_, nameOfNN_, sensor_list_, sensor_info_, metrics_, metric_value_, dataset_, date_, creator_,
                        location_, reference_, description_, category_, input_comment_, output_comment_, runtime_):
    with open(path_, "rb") as f:
        buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)

    interpreter = tf.lite.Interpreter(model_path=path_)
    interpreter.allocate_tensors()

    idOfNN = str(uuid.uuid4())
    # Copy the model to the destination repo and rename it with uuid
    base, extension = os.path.splitext(path_)
    # Regenerate a uuid if an existing model has the same uuid
    while True:
        newPath = os.path.join(os.getcwd(), destination, idOfNN + extension)
        if not os.path.isfile(newPath):
            shutil.copy(path_, newPath)
            break
        idOfNN = str(uuid.uuid4())

    # ----------- semantic handling 1 -----------
    neuralNetwork = URIRef("http://tinyml-schema.org/neuralnetwork/" + idOfNN)

    inputOfNN = URIRef("http://tinyml-schema.org/neuralnetwork/input_" + idOfNN)
    outputOfNN = URIRef("http://tinyml-schema.org/neuralnetwork/output_" + idOfNN)
    metricOfNN = URIRef("http://tinyml-schema.org/neuralnetwork/metric_1_" + idOfNN)
    procedureFeature_1 = URIRef("http://tinyml-schema.org/neuralnetwork/procedureFeature_1_" + idOfNN)
    condition_1 = URIRef("http://tinyml-schema.org/neuralnetwork/condition_1_" + idOfNN)
    procedureFeature_2 = URIRef("http://tinyml-schema.org/neuralnetwork//procedureFeature_2_" + idOfNN)
    condition_2 = URIRef("http://tinyml-schema.org/neuralnetwork/condition_2_" + idOfNN)

    # NN model metadata
    g.add((neuralNetwork, RDF.type, nnet.NeuralNetwork))
    g.add((neuralNetwork, nnet.trainingDataset, URIRef(dataset_)))
    g.add((neuralNetwork, SDO.name, Literal(nameOfNN_)))
    g.add((neuralNetwork, SDO.identifier, Literal(idOfNN)))
    g.add((neuralNetwork, SDO.citation, URIRef(reference_)))
    g.add((neuralNetwork, SDO.dateCreated, Literal(date_)))
    g.add((neuralNetwork, SDO.creator, Literal(creator_)))
    g.add((neuralNetwork, SDO.description, Literal(description_)))
    g.add((neuralNetwork, SDO.runtimePlatform, Literal(runtime_)))
    g.add((neuralNetwork, SDO.codeRepository, URIRef(location_)))
    # Tensorflot lite model has only one subgraph
    graph = model.Subgraphs(0)
    # Calculate MACs
    total_mac = calc_mac(graph, model)
    g.add((neuralNetwork, nnet.hasMultiplyAccumulateOps, Literal(total_mac)))

    # Add sensor information
    def addHardware(code):
        # add hardware information to NN
        # only a few common sensors are implemented
        CODE2HARDWARE = {
            0: sosa_extend.Camera,
            1: sosa_extend.Microphone,
            2: sosa_extend.Accelerometer,
            3: sosa_extend.Gyroscope,
            4: sosa_extend.Thermometer,
            5: sosa_extend.OtherSensor,
        }
        return CODE2HARDWARE[code]

    for i, value in enumerate(sensor_list_):
        sensorOfNN = URIRef("http://tinyml-schema.org/neuralnetwork/sensor" + "_" + "{}".format(i + 1) + "_" + idOfNN)
        g.add((sensorOfNN, RDF.type, addHardware(value)))
        g.add((sensorOfNN, RDF.type, SDO.Sensor))
        g.add((sensorOfNN, sosa_extend.hasSensorInfo, Literal(sensor_info_)))
        g.add((sensorOfNN, ssn_extend.provideInput, inputOfNN))

    # Add input and output information of NN
    g.add((neuralNetwork, SSN.hasInput, inputOfNN))
    g.add((inputOfNN, RDF.type, nnet.NetworkInput))
    g.add((inputOfNN, nnet.hasInputInfo, Literal(input_comment_)))

    g.add((neuralNetwork, SSN.hasOutput, outputOfNN))
    g.add((outputOfNN, RDF.type, nnet.NetworkOutput))
    g.add((outputOfNN,nnet.hasOutputInfo, Literal(output_comment_)))

    # Add metric of NN
    def addMetric(code):
        # add metric information to NN
        # only a few common metrics are implemented
        CODE2METRIC = {
            0: nnet.Top_1_accuracy,
            1: nnet.Top_5_accuracy,
            2: nnet.Other_metric,
        }
        g.add((metricOfNN, RDF.type, CODE2METRIC[code]))
    addMetric(metrics_)
    g.add((metricOfNN, nnet.hasMetricValue, Literal(metric_value_)))
    g.add((metricOfNN, RDF.type, nnet.Metric))
    g.add((neuralNetwork, nnet.hasMetric, metricOfNN))

    # Add category of NN
    def addCategory(code):
        # add metric information to NN
        # only a few common metrics are implemented
        CODE2CATEGORY = {
            0: nnet.Classification,
            1: nnet.ObjectDetection,
            2: nnet.FeatureExtraction,
            3: nnet.Unsupervised,
            4: nnet.OtherCategory,
        }
        g.add((neuralNetwork, nnet.hasCategory, CODE2CATEGORY[code]))
    addCategory(category_)

    # Add flash requirement of NN.
    g.add((condition_1, RDF.type, s3n.Memory))
    g.add((condition_1, RDF.type, s3n_extend.Flash))
    g.add((condition_1, SDO.unitCode, om.kilobyte))
    g.add((condition_1, RDFS.label, Literal("Flash requirement.")))
    g.add((procedureFeature_1, RDF.type, s3n.ProcedureFeature))
    g.add((procedureFeature_1, RDFS.label, Literal("procedureFeature_1")))
    g.add((procedureFeature_1, ssn_system.inCondition, condition_1))
    size = os.path.getsize(path_)
    g.add((condition_1, SDO.minValue, Literal(size / 1000)))
    g.add((neuralNetwork, s3n.hasProcedureFeature, procedureFeature_1))
    # Add RAM requirement of NN.
    g.add((condition_2, RDF.type, s3n.Memory))
    g.add((condition_2, RDF.type, s3n_extend.RAM))
    g.add((condition_2, SDO.unitCode, om.kilobyte))
    g.add((condition_2, RDFS.label, Literal("RAM requirement.")))
    g.add((procedureFeature_2, RDF.type, s3n.ProcedureFeature))
    g.add((procedureFeature_2, RDFS.label, Literal("procedureFeature_2")))
    g.add((procedureFeature_2, ssn_system.inCondition, condition_2))
    tensorSize = int(os.popen("./find-arena-size {}".format(path_)).read())
    g.add((condition_2, SDO.minValue, Literal(tensorSize / 1000)))
    g.add((neuralNetwork, s3n.hasProcedureFeature, procedureFeature_2))

    print("############################################ Model Information ############################################")
    print("The size of the model: {} bytes".format(size))
    print("The memory requirement of the model: {} bytes".format(tensorSize))
    print("Model version : {}".format(model.Version()))
    # Strings are binary format, need to decode.
    # Description is useful when exchanging models.
    print("Model description : {}".format(model.Description().decode()))
    # How many operator types in this model.
    print("Operator types: {}".format(model.OperatorCodesLength()))
    # A model may have multiple subgraphs.
    print("Subgraph number: {}".format(model.SubgraphsLength()))
    # How many tensor buffer.
    print("Tensor buffer number: {}".format(model.BuffersLength()))

    # Start processing layer-specific information
    # Select a subgraph, tflite model has only one subgraph
    for i in range(model.SubgraphsLength()):

        print("###################################### Subgraph {} ######################################".format(i + 1))
        graph = model.Subgraphs(i)
        # Tensors in the subgraph are represented by index description.

        # Operators in the subgraph.
        print("Operator number: {}".format(graph.OperatorsLength()))
        print("Input length: {}".format(graph.InputsLength()))
        print("Output length: {}".format(graph.OutputsLength()))
        print("Input dim: {}".format(graph.InputsAsNumpy()[0]))
        print("Output dim: {}".format(graph.OutputsAsNumpy()[0]))
        total_mac = calc_mac(graph, model)
        print("Total mac: {}".format(total_mac))

        # For each operator node:
        for j in range(graph.OperatorsLength()):
            print("############################## Operator {} ##############################".format(j + 1))
            op = graph.Operators(j)
            # Operator Type is also stored as index, which can obtain from `Model` object.
            print("The {} th operator is: {}".format(j + 1, tflite.opcode2name(
                model.OperatorCodes(op.OpcodeIndex()).BuiltinCode())))
            # The inputs are: data, weight and bias
            print("Operator input length: {}".format(op.InputsLength()))
            print("Operator output length: {}".format(op.OutputsLength()))
            # The data of first operator is input of the model
            print("Operator inputs: {}".format(op.Inputs(0)))
            print("Operator outputs: {}".format(op.Outputs(0)))

            # Parse the Table of options.
            op_opt = op.BuiltinOptions()

            def addOptionByCode(code):
                # Only Several Options are implemented Here
                CODE2OPTION = {
                    0: tflite.AddOptions(),
                    2: tflite.ConcatenationOptions(),
                    3: tflite.Conv2DOptions(),
                    4: tflite.DepthwiseConv2DOptions(),
                    6: tflite.DequantizeOptions(),
                    9: tflite.FullyConnectedOptions(),
                    18: tflite.MulOptions(),
                    22: tflite.ReshapeOptions(),
                    25: tflite.SoftmaxOptions(),
                    34: tflite.PadOptions(),
                    41: tflite.SubOptions(),
                    49: tflite.SplitOptions(),
                    83: tflite.PackOptions(),
                    88: tflite.UnpackOptions(),
                    97: tflite.ResizeNearestNeighborOptions(),
                    102: tflite.SplitVOptions(),
                    114: tflite.QuantizeOptions(),
                }
                if code in CODE2OPTION:
                    return CODE2OPTION[code]
                else:
                    CRED = '\033[91m'
                    CEND = '\033[0m'
                    print(CRED + "No Option Found for This Operator!" + CEND)
                    return False

            opt = addOptionByCode(model.OperatorCodes(op.OpcodeIndex()).BuiltinCode())

            quantized = True
            hasTensor = False
            hasActivation = False

            if (op_opt is not None) and opt:
                opt.Init(op_opt.Bytes, op_opt.Pos)
                # Reshape and Softmax operator has to be handled differently
                if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() in [tflite.BuiltinOperator.RESHAPE, tflite.BuiltinOperator.SOFTMAX]:
                    print("Activation function: {}".format(activationcode2name(0)))
                else:
                    # Further check activation function type if there exists.
                    hasTensor = True
                    if hasattr(opt, 'FusedActivationFunction'):
                        hasActivation = True
                        print("Activation function: {}".format(activationcode2name(opt.FusedActivationFunction())))
                    else:
                        hasActivation = False
                        print("Activation function: {}".format(activationcode2name(0)))

                    # The three dimention of inputs are: data, weight and bias
                    # Check the weight tensor
                    print("######### weight tensor #########")
                    tensor_index = op.Inputs(1)
                    # use `graph.Tensors(index)` to get the tensor object.
                    tensor = graph.Tensors(tensor_index)
                    # print("shape length of the tensor: {}".format(tensor.ShapeLength()))
                    print("Tensor shape: {}".format(tensor.ShapeAsNumpy()))
                    # print("input dimension: {}".format(tensor.ShapeAsNumpy()))
                    # print("buffer length: {}".format(buf.DataLength()))

                    # Check if a layer is quantized: if it is not quantized, the variable should be type of int
                    if isinstance(tensor.Quantization().ScaleAsNumpy(), np.ndarray):
                        print("Quantized layer: Yes")
                    else:
                        print("Quantized layer: No")
                        quantized = False
                    print("Tensor Name: {}".format(tensor.Name().decode("utf-8")))

            # ----------- semantic handling 3 -----------
            def addLayer(layer, code):
                # add layer type information to each layer
                # only a few common operators are implemented
                CODE2LAYER = {
                    0: nnet.Add,
                    1: nnet.AvgPool2D,
                    2: nnet.Concatenation,
                    3: nnet.Conv2D,
                    4: nnet.DepthwiseConv2D,
                    6: nnet.Dequantize,
                    9: nnet.FullyConnected,
                    14: nnet.Logistic,
                    17: nnet.MaxPool2D,
                    18: nnet.Mul,
                    22: nnet.Reshape,
                    25: nnet.Softmax,
                    28: nnet.Tanh,
                    34: nnet.Pad,
                    40: nnet.Mean,
                    41: nnet.Sub,
                    49: nnet.Split_,
                    83: nnet.Pack,
                    80: nnet.FakeQuant,
                    88: nnet.Unpack,
                    97: nnet.ResizeNearestNeighbor,
                    102: nnet.SplitV,
                    114: nnet.Quantize,
                }
                if code in CODE2LAYER:
                    g.add((layer, nnet.hasType, CODE2LAYER[code]))
                else:
                    raise ValueError("Unknown layercode %d, might be a custom layer." % code)

            def addQuantization(code):
                # Reference: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/lite/schema/schema.fbs/#L32
                CODE2DATATYPE = {
                    0: nnet.Float32,
                    1: nnet.Float16,
                    2: nnet.Int32,
                    3: nnet.Uint8,
                    4: nnet.Int64,
                    5: nnet.String,
                    6: nnet.Bool,
                    7: nnet.Int16,
                    8: nnet.Complex64,
                    9: nnet.Int8,
                    10: nnet.Float64,
                    11: nnet.Complex128,
                }
                if code in CODE2DATATYPE:
                    return CODE2DATATYPE[code]
                else:
                    raise ValueError("Unknown datatype code %d, might be a custom operator." % code)

            def addTrainable(layer, quantization_flag):
                # add quantization information to each layer
                # if quantization_flag:
                #     g.add((layer, nnet.isTrainable, SDO.false))
                # else:
                #     g.add((layer, nnet.isTrainable, SDO.true))
                # Add data type to the tensor of each node/layer
                if hasTensor:
                    g.add((layer, nnet.hasQuantization, addQuantization(tensor.Type())))

            def addActivation(layer, code):
                # add activation information to semantic description.
                # Only a few common activation are implemented
                CODE2ACTIVATION = {
                    1: nnet.Relu,
                    2: nnet.Relu_n1_to_1,
                    3: nnet.Relu6,
                    4: nnet.Tanh,
                    5: nnet.Sign_bit,
                }
                if code in CODE2ACTIVATION:
                    g.add((layer, nnet.hasActivation, CODE2ACTIVATION[code]))
                else:
                    CRED = '\033[91m'
                    CEND = '\033[0m'
                    print(CRED + "No Activation Found for This Layer!" + CEND)

            def addCommonInfo(layer, input_layer=False, output_layer=False):
                g.add((layer, RDF.type, nnet.Layer))
                addTrainable(layer, quantized)
                addLayer(layer, model.OperatorCodes(op.OpcodeIndex()).BuiltinCode())

                if input_layer:
                    g.add((layer, nnet.shapeIn, Literal(interpreter.get_input_details()[0]['shape'])))
                if output_layer:
                    g.add((layer, nnet.shapeOut, Literal(interpreter.get_output_details()[0]['shape'])))

                if hasActivation:
                    addActivation(layer, opt.FusedActivationFunction())
                else:
                    addActivation(layer, 0)

            if j == 0:
                # First layer is input layer
                inputLayer = URIRef("http://tinyml-schema.org/neuralnetwork/inputLayer_" + idOfNN)
                g.add((neuralNetwork, nnet.inputLayer, inputLayer))
                addCommonInfo(inputLayer, input_layer=True)
                g.add((inputLayer, nnet.hasIndex, Literal(j+1)))
            elif j == (graph.OperatorsLength() - 1):
                # Last layer is output layer
                outputLayer = URIRef("http://tinyml-schema.org/neuralnetwork/outputLayer_" + idOfNN)
                g.add((neuralNetwork, nnet.outputLayer, outputLayer))
                addCommonInfo(outputLayer, output_layer=True)
                g.add((outputLayer, nnet.hasIndex, Literal(j+1)))
            else:
                middleLayer = URIRef("http://tinyml-schema.org/neuralnetwork/middleLayer_" + "{}_".format(j) + idOfNN)
                g.add((neuralNetwork, nnet.middleLayer, middleLayer))
                g.add((middleLayer, nnet.hasIndex, Literal(j)))
                addCommonInfo(middleLayer)
                # g.add((middleLayer, RDFS.label, Literal("middleLayer_" + idOfNN + "_{}".format(j))))
                g.add((middleLayer, nnet.hasIndex, Literal(j)))
            # --------------------------------------


if __name__ == "__main__":

    # THe url where the model will be hosted
    original = r"collected_models"
    destination = r"model_repo"
    # Specify two folders for the models to be parsed and the model to be hosted
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_model_dir = os.path.join(cur_dir, original)
    model_repo_dir = os.path.join(cur_dir, destination)
    print(test_model_dir)
    workbook = pd.read_excel(os.path.join(cur_dir, "Models_Information.xlsx"), sheet_name=0)

    headers = [col for col in workbook.columns]
    model_count = 0

    for path in sorted(glob.glob(os.path.join(test_model_dir, "*.tflite"))):
        print("Model found: {}".format(path))

        base = os.path.splitext(os.path.basename(path))[0]

        # Ask user extra information regarding the model
        nameOfNN = workbook.loc[model_count][headers[0]]

        # One can specify the type of the sensor which is used to collect data for NN training, as well as specific sensor setting
        sensor = workbook.loc[model_count][headers[1]]
        if not isinstance(sensor, int):
            sensor_list = set(map(int, sensor.split(",")))
        else:
            sensor_list = [sensor]
        sensor_info = workbook.loc[model_count][headers[2]]

        input_comment = workbook.loc[model_count][headers[3]]

        output_comment = workbook.loc[model_count][headers[4]]

        metrics = workbook.loc[model_count][headers[5]]
        metric_value = workbook.loc[model_count][headers[6]]

        category = workbook.loc[model_count][headers[7]]

        location = workbook.loc[model_count][headers[8]]
        dataset = workbook.loc[model_count][headers[9]]
        reference = workbook.loc[model_count][headers[10]]

        # If no specific time is provided, use the modification time of the file
        if workbook.loc[model_count][headers[11]] == "None":
            modTimesinceEpoc = os.path.getmtime(path)
            modificationTime = time.strftime('%Y-%m-%d', time.localtime(modTimesinceEpoc))
            date = modificationTime
        else:
            date = workbook.loc[model_count][headers[11]]

        creator = workbook.loc[model_count][headers[12]]

        description = workbook.loc[model_count][headers[13]]

        runtime_platform = workbook.loc[model_count][headers[14]]

        tflite2semantic(path, nameOfNN, sensor_list, sensor_info, metrics, metric_value, dataset, date, creator,
                        location, reference, description, category, input_comment, output_comment, runtime_platform)
        model_count += 1
        # print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(nameOfNN, sensor_list, sensor_info, input_comment, output_comment, metrics, metric_value, category, location, dataset, reference, date, creator, description, runtime_platform))

    print("###################################Semantic Description#######################################")
    print(g.serialize(format='turtle').decode())
    # Save the ttl file
    with open(os.path.join(os.getcwd(), "semantic_schema", "tflite_models.ttl"), 'w') as f:
        print(g.serialize(format='turtle').decode(), file=f)
