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

# Reference: https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/lite/schema/schema.fbs/#L344

# Create namespace for not-supported namespace by rdflib
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
nnet = Namespace("http://ureasoner.org/networkschema#")
s3n = Namespace("http://w3id.org/s3n/")
ssn_system = Namespace("http://www.w3.org/ns/ssn/systems/")
demo = Namespace("http://ureasoner.org/tinyml-demo#")

# Create a Graph
g = Graph()
# Name binding
nm = g.namespace_manager
nm.bind("sosa", SOSA)
nm.bind("nnet", nnet)
nm.bind("schema", SDO)
nm.bind("s3n", s3n)
nm.bind("ssn", SSN)
nm.bind("ssn-system", ssn_system)
nm.bind("demo", demo)


def tflite2semantic(path_, nameOfNN_, hardware_list_, hardware_info_, metrics_, metric_value_, dataset_, date_, creator_,
                        location_, reference_, description_, category_, input_comment_, output_comment_):
    with open(path_, "rb") as f:
        buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)

    interpreter = tf.lite.Interpreter(model_path=path_)
    interpreter.allocate_tensors()

    idOfNN = str(uuid.uuid4())
    # Copy the model to the destination and rename it with uuid
    base, extension = os.path.splitext(path_)
    # Regenerate a uuid if an existing model has the same uuid
    while True:
        newPath = os.path.join(os.getcwd(), destination, idOfNN + extension)
        if not os.path.isfile(newPath):
            shutil.copy(path_, newPath)
            break
        idOfNN = str(uuid.uuid4())

    # ----------- semantic handling 1 -----------
    neuralNetwork = URIRef("http://example.org/neuralnetwork/" + idOfNN)

    inputOfNN = URIRef("http://example.org/neuralnetwork/input_" + idOfNN)
    outputOfNN = URIRef("http://example.org/neuralnetwork/output_" + idOfNN)
    metricOfNN = URIRef("http://example.org/neuralnetwork/metric_1_" + idOfNN)
    procedureFeature_1 = URIRef("http://example.org/neuralnetwork/procedureFeature_1_" + idOfNN)
    condition_1 = URIRef("http://example.org/neuralnetwork/condition_1_" + idOfNN)
    procedureFeature_2 = URIRef("http://example.org/neuralnetwork/procedureFeature_2_" + idOfNN)
    condition_2 = URIRef("http://example.org/neuralnetwork/condition_2_" + idOfNN)

    # NN model data properties
    g.add((neuralNetwork, RDF.type, nnet.NeuralNetwork))
    g.add((neuralNetwork, nnet.location, URIRef(location_)))
    g.add((neuralNetwork, nnet.dataset, URIRef(dataset_)))
    g.add((neuralNetwork, nnet.hasTitle, Literal(nameOfNN_)))
    g.add((neuralNetwork, nnet.hasID, Literal(idOfNN)))
    g.add((neuralNetwork, nnet.reference, URIRef(reference_)))
    g.add((neuralNetwork, nnet.created, Literal(date_)))
    g.add((neuralNetwork, nnet.creator, Literal(creator_)))
    g.add((neuralNetwork, nnet.hasDescription, Literal(description_)))
    graph = model.Subgraphs(0)
    total_mac = calc_mac(graph, model)
    g.add((neuralNetwork, nnet.hasMultiplyAccumulateOps, Literal(total_mac)))

    # Object property: hardware
    def addHardware(code):
        # add hardware information to NN
        # only a few common sensors are implemented
        CODE2HARDWARE = {
            0: nnet.Camera,
            1: nnet.Microphone,
            2: nnet.Accelerometer,
            3: nnet.Gyroscope,
            4: nnet.Thermometer,
            5: nnet.OtherSensor,
        }
        return CODE2HARDWARE[code]

    for i, value in enumerate(hardware_list_):
        hardwareOfNN = URIRef("http://example.org/neuralnetwork/hardware" + "_" + "{}".format(i + 1) + "_" + idOfNN)
        g.add((hardwareOfNN, RDF.type, addHardware(value)))
        g.add((hardwareOfNN, RDF.type, nnet.Hardware))
        g.add((hardwareOfNN, nnet.hasHardwareInfo, Literal(hardware_info_)))
        g.add((neuralNetwork, nnet.hasHardwareReference, hardwareOfNN))

    # Object property: input and output of NN
    g.add((neuralNetwork, SSN.hasInput, inputOfNN))
    g.add((inputOfNN, RDF.type, nnet.NetworkInput))
    g.add((inputOfNN, nnet.hasInputInfo, Literal(input_comment_)))

    g.add((neuralNetwork, SSN.hasOutput, outputOfNN))
    g.add((outputOfNN, RDF.type, nnet.NetworkOutput))
    g.add((outputOfNN,nnet.hasOutputInfo, Literal(output_comment_)))

    # Object property: Metric of NN
    def addMetric(code):
        # add metric information to NN
        # only a few common metrics are implemented
        CODE2METRIC = {
            0: nnet.top_1_accuracy,
            1: nnet.top_5_accuracy,
            2: nnet.other_metric,
        }
        g.add((metricOfNN, RDF.type, CODE2METRIC[code]))
    addMetric(metrics_)
    g.add((metricOfNN, nnet.hasMetricValue, Literal(metric_value_)))
    g.add((metricOfNN, RDF.type, nnet.Metric))
    g.add((neuralNetwork, nnet.hasMetric, metricOfNN))

    # Object property: category of NN
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

    # Object property: Flash requirement of NN.
    g.add((condition_1, RDF.type, s3n.Memory))
    g.add((condition_1, RDF.type, demo.Flash))
    g.add((condition_1, SDO.unitCode, Literal("om.kilobyte")))
    g.add((condition_1, RDFS.label, Literal("Flash requirement.")))
    g.add((procedureFeature_1, RDF.type, s3n.ProcedureFeature))
    g.add((procedureFeature_1, RDFS.label, Literal("procedureFeature_1")))
    g.add((procedureFeature_1, ssn_system.inCondition, condition_1))
    size = os.path.getsize(path_)
    g.add((condition_1, SDO.minValue, Literal(size / 1000)))
    g.add((neuralNetwork, s3n.hasProcedureFeature, procedureFeature_1))
    # # Object property: Ram requirement of NN.
    g.add((condition_2, RDF.type, s3n.Memory))
    g.add((condition_2, RDF.type, demo.RAM))
    g.add((condition_2, SDO.unitCode, Literal("om.kilobyte")))
    g.add((condition_2, RDFS.label, Literal("Ram requirement.")))
    g.add((procedureFeature_2, RDF.type, s3n.ProcedureFeature))
    g.add((procedureFeature_2, RDFS.label, Literal("procedureFeature_2")))
    g.add((procedureFeature_2, ssn_system.inCondition, condition_2))
    # print(path_)
    # print(os.popen("./find-arena-size {}".format(path_)).read())
    tensorSize = int(os.popen("./find-arena-size {}".format(path_)).read())
    # tensorSize = 222
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

    # Select a subgraph
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

                    # Check if a layer is quantized: if it is not quantized, it should be type of int
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
                    0: nnet.add,
                    1: nnet.avgPool2D,
                    2: nnet.concatenation,
                    3: nnet.conv2D,
                    4: nnet.depthwiseConv2D,
                    6: nnet.dequantize,
                    9: nnet.fullyConnected,
                    14: nnet.logistic,
                    17: nnet.maxPool2D,
                    18: nnet.mul,
                    22: nnet.reshape,
                    25: nnet.softmax,
                    28: nnet.tanh,
                    34: nnet.pad,
                    40: nnet.mean,
                    41: nnet.sub,
                    49: nnet.split_,
                    83: nnet.pack,
                    80: nnet.fake_quant,
                    88: nnet.unpack,
                    97: nnet.resizeNearestNeighbor,
                    102: nnet.split_v,
                    114: nnet.quantize,
                }
                if code in CODE2LAYER:
                    g.add((layer, nnet.hasType, CODE2LAYER[code]))
                else:
                    raise ValueError("Unknown layercode %d, might be a custom layer." % code)

            def addQuantization(code):
                # Reference: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/lite/schema/schema.fbs/#L32
                CODE2DATATYPE = {
                    0: nnet.float32,
                    1: nnet.float16,
                    2: nnet.int32,
                    3: nnet.uint8,
                    4: nnet.int64,
                    5: nnet.string,
                    6: nnet.bool,
                    7: nnet.int16,
                    8: nnet.complex64,
                    9: nnet.int8,
                    10: nnet.float64,
                    11: nnet.complex128,
                }
                if code in CODE2DATATYPE:
                    return CODE2DATATYPE[code]
                else:
                    raise ValueError("Unknown datatype code %d, might be a custom operator." % code)

            def addTrainable(layer, quantization_flag):
                # add quantization and trainable information to the each layer
                if quantization_flag:
                    g.add((layer, nnet.isTrainable, SDO.false))
                else:
                    g.add((layer, nnet.isTrainable, SDO.true))
                # Add data type to the tensor of each node/layer
                if hasTensor:
                    g.add((layer, nnet.hasQuantization, addQuantization(tensor.Type())))

            def addActivation(layer, code):
                # add activation information to semantic description.
                # Only a few common activation are implemented
                CODE2ACTIVATION = {
                    1: nnet.relu,
                    2: nnet.relu_n1_to_1,
                    3: nnet.relu6,
                    4: nnet.tanh,
                    5: nnet.sign_bit,
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
                inputLayer = URIRef("http://example.org/neuralnetwork/inputLayer_" + idOfNN)
                g.add((neuralNetwork, nnet.inputLayer, inputLayer))
                addCommonInfo(inputLayer, input_layer=True)
                g.add((inputLayer, nnet.hasIndex, Literal(j+1)))
            elif j == (graph.OperatorsLength() - 1):
                # Last layer is output layer
                outputLayer = URIRef("http://example.org/neuralnetwork/outputLayer_" + idOfNN)
                g.add((neuralNetwork, nnet.outputLayer, outputLayer))
                addCommonInfo(outputLayer, output_layer=True)
                g.add((outputLayer, nnet.hasIndex, Literal(j+1)))
            else:
                middleLayer = URIRef("http://example.org/neuralnetwork/middleLayer_" + "{}_".format(j) + idOfNN)
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
    for path in sorted(glob.glob(os.path.join(test_model_dir, "*.tflite"))):
        print("Model found: {}".format(path))

        base = os.path.splitext(os.path.basename(path))[0]

        # Ask user extra information regarding the model
        nameOfNN = (input("The Name of the neural network: ") or base)

        # One can specify the type of the hardware which is used to collect data for NN training, as well as specific hardware setting
        hardware = input(
            "Hardware for data acquisition (0=Camera, 1=Microphone, 2=Accelerometer, 3=Gyroscope, 4=Thermometer, 5=OtherSensor, separated by comma): ")
        while len(hardware) < 1 or not set(map(int, hardware.split(","))).issubset([i for i in range(6)]):
            hardware = (input("Hardware for data acquisition (0=Camera, 1=Microphone, 2=Accelerometer, 3=Gyroscope, 4=Thermometer, 5=OtherSensor, separated by comma): "))
        hardware_list = set(map(int, hardware.split(",")))
        hardware_info = (input("Hardware information (such as model and setting): ") or "None")

        input_comment = (input("Information about the neural network INPUT (such as dimension, scaling): ") or "No comment")

        output_comment = (input("Information about the neural network OUTPUT (such as category for each dimension): ") or "No comment")

        metrics = input("Metrics for evaluating the NN (0=top_1_acc, 1=top_5_acc, 2=other_metric): ")
        while len(metrics) < 1 or int(metrics) not in [i for i in range(3)]:
            metrics = input("Metrics for evaluating the NN (0=top_1_acc, 1=top_5_acc, 2=other_metric): ")
        metrics = int(metrics)
        metric_value = (input("Metric value: ") or 0)

        category = input(
            "Category of the NN (0=Classification, 1=ObjectDetection, 2=FeatureExtraction, 3=Unsupervised, 4=Other): ")
        while len(category) < 1 or int(category) not in [i for i in range(5)]:
            category = input(
                "Category of the NN (0=Classification, 1=ObjectDetection, 2=FeatureExtraction, 3=Unsupervised, 4=Other): ")
        category = int(category)

        location = (input("URL for storing the model: ") or "https://github.com/Haoyu-R/TinyML-Research-Symposium-2022/tree/main/model_repo")
        dataset = (input("URL of Dataset for training NN: ") or "None")
        reference = (input("URL for the reference of the model for more information: ") or "None")

        # If no specific time is provided, use the modification time of the file
        modTimesinceEpoc = os.path.getmtime(path)
        modificationTime = time.strftime('%Y-%m-%d', time.localtime(modTimesinceEpoc))
        date = (input("When is the model created (%Y-%m-%d): ") or modificationTime)

        creator = (input("Who created the model: ") or "None")

        description = (input("Description about the model: ") or "None")

        tflite2semantic(path, nameOfNN, hardware_list, hardware_info, metrics, metric_value, dataset, date, creator,
                        location, reference, description, category, input_comment, output_comment)
        print()
    print("###################################Semantic Description#######################################")
    print(g.serialize(format='turtle').decode())
    # Save the ttl file
    with open(os.path.join(os.getcwd(), "semantic_schema", "tflite_models.ttl"), 'w') as f:
        print(g.serialize(format='turtle').decode(), file=f)

