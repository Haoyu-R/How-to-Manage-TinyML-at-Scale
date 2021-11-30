import tflite

def calc_mac(graph, model):
    # Estimate the number of MAC of a model, currently only fullyConnected/Conv_2D/DEPTHWISE_CONV_2D are counted
    print()
    # print functions
    _dict_builtin_op_code_to_name = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if type(v) == int}

    def print_header():
        print("%-18s | MAC" % ("OP_NAME"))
        print("------------------------------")

    def print_mac(op_code_builtin, mac):
        print("%-18s | %.1f" % (_dict_builtin_op_code_to_name[op_code_builtin], mac))

    def print_none(op_code_builtin):
        print("%-18s | <IGNORED>" % (_dict_builtin_op_code_to_name[op_code_builtin]))

    def print_footer(total_mac):
        print("------------------------------")
        print("Total: %.1f mac" % (total_mac))

    total_mac = 0.0
    print_header()
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        op_code_builtin = op_code.BuiltinCode()
        op_opt = op.BuiltinOptions()

        mac = 0.0
        if op_code_builtin == tflite.BuiltinOperator.CONV_2D:
            # input shapes: in, weight, bias
            in_shape = graph.Tensors(op.Inputs(0)).ShapeAsNumpy()
            filter_shape = graph.Tensors(op.Inputs(1)).ShapeAsNumpy()
            bias_shape = graph.Tensors(op.Inputs(2)).ShapeAsNumpy()
            # output shape
            out_shape = graph.Tensors(op.Outputs(0)).ShapeAsNumpy()
            # ops options
            opt = tflite.Conv2DOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            # opt.StrideH()

            # mac. 2x means mul(1)+add(1). 2x if you calculate flops
            # refer to https://github.com/AlexeyAB/darknet/src/convolutional_layer.c `l.blopfs =`
            mac = out_shape[1] * out_shape[2] * filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]
            print_mac(op_code_builtin, mac)

        elif op_code_builtin == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
            in_shape = graph.Tensors(op.Inputs(0)).ShapeAsNumpy()
            filter_shape = graph.Tensors(op.Inputs(1)).ShapeAsNumpy()
            out_shape = graph.Tensors(op.Outputs(0)).ShapeAsNumpy()
            # mac
            mac = out_shape[1] * out_shape[2] * filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]
            print_mac(op_code_builtin, mac)
        elif op_code_builtin == tflite.BuiltinOperator.FULLY_CONNECTED:
            out_shape = graph.Tensors(op.Inputs(1)).ShapeAsNumpy()[0]
            in_shape = graph.Tensors(op.Inputs(1)).ShapeAsNumpy()[1]
            mac = in_shape * out_shape
            print_mac(op_code_builtin, mac)
        else:
            print_none(op_code_builtin)

        total_mac += mac
    # print_footer(total_mac)
    return total_mac


def activationcode2name(code):
    # Return the activation function name given code
    # Reference: https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/lite/schema/schema.fbs/#L344
    ACTIVATIONCODE2NAME = {
        0: 'NONE',
        1: 'RELU',
        2: 'RELU_N1_TO_1',
        3: 'RELU6',
        4: 'TANH',
        5: 'SIGN_BIT',
    }
    if code in ACTIVATIONCODE2NAME:
        return ACTIVATIONCODE2NAME[code]
    else:
        raise ValueError("Unknown activationcode %d, might be a custom operator." % code)


