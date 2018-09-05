import argparse
import code
import sys

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2, attr_value_pb2, node_def_pb2
from tensorflow.python.framework import tensor_shape

def process_shapes(graph_def, input_node, input_shape_string):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    outputs = [n.outputs[0] for n in graph.get_operations() if n.type == 'Shape']
    input_shape = [int(x.strip()) for x in input_shape_string.split(',')]
    if len(input_shape) != 4:
        print('Invalid dimensions: Input shape should be 4 comma-separated integers')

    main_input = graph.get_operation_by_name(input_node).outputs[0]
    feed_dict = {main_input: np.random.rand(input_shape[0],input_shape[1],input_shape[2],input_shape[3])}
    with tf.Session(graph=graph) as sess:
        rawOut = sess.run(outputs, feed_dict)

    new_model = graph_pb2.GraphDef()
    with tf.Session(graph=graph) as sess:
        for n in sess.graph_def.node:
            if n.name == input_node:
                nn = new_model.node.add()
                nn.op = 'Placeholder'
                nn.name = input_node
                nn.attr['dtype'].CopyFrom(n.attr['dtype'])
                shape = input_shape
                nn.attr["shape"].CopyFrom(attr_value_pb2.AttrValue(shape=tensor_shape.TensorShape(shape).as_proto()))
            elif n.op == 'Shape':
                nn = new_model.node.add()
                nn.name = n.name
                nn.op = 'Const'
                nn.attr['dtype'].type = 3
                nn.attr['value'].tensor.dtype = 3
                shape = rawOut[next(i for i,n in enumerate(outputs) if n.op.name == nn.name)]
                temp = nn.attr['value'].tensor.tensor_shape.dim.add()
                temp.size = shape.size
                nn.attr['value'].tensor.tensor_content = np.array(shape).astype(np.int32).tobytes()
            # Handle redundant Reshape ops
            elif n.op == 'Reshape':
                nn = new_model.node.add()
                nn.CopyFrom(n)
                op = tf.import_graph_def(new_model, return_elements=[n.name], name="")[0]
                if op.inputs[0].shape.as_list() == op.outputs[0].shape.as_list():
                    temp = node_def_pb2.NodeDef()
                    temp.op = 'Identity'
                    temp.name = nn.name
                    temp.input.extend([nn.input[0]])
                    temp.attr['T'].CopyFrom(nn.attr['T'])
                    new_model.node.pop()
                    new_model.node.extend([temp])    
            else:
                nn = new_model.node.add()
                nn.CopyFrom(n)
    return new_model

parser = argparse.ArgumentParser(description='Converts shape ops into constants to further fold child ops.')
parser.add_argument('-p', '--input-graph', help='Required argument: input_model.pb')
parser.add_argument('-o', '--output-graph', default='constantified_model.pb', help='Output protobuf')
parser.add_argument('-in', '--input-node', required=True, default=None, help='Name of the input node')
parser.add_argument('-is', '--input-shape', default='1,224,224,3', help='4-D shape of input tensor (comma-separated), e.g. 1,227,227,3')
args = parser.parse_args()

model = args.input_graph
dest = args.output_graph
input_node = args.input_node
input_shape_string = args.input_shape

with tf.gfile.GFile(model, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

output_def = process_shapes(graph_def, input_node, input_shape_string)

with tf.gfile.GFile(dest, "wb") as f:
    f.write(output_def.SerializeToString())
