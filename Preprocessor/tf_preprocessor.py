l#!/usr/bin/env python3

import argparse
import code
import sys

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import tensor_shape

# from constantify_shapes import process_shapes as CS
import utility_library
from graph_transform import TransformGraph as transform_graph

DEFAULT_TRANSFORMS = 'fold_constants fold_batch_norms fold_old_batch_norms merge_duplicate_nodes remove_nodes(op=Identity op=CheckNumerics op=Assert) strip_unused_nodes remove_attribute(attribute_name=_class) sort_by_execution_order'
POST_CS_TRANSFORMS = 'remove_control_dependencies remove_nodes(op=Identity) fold_constants merge_duplicate_nodes remove_attribute(attribute_name=_class) sort_by_execution_order'

def summarize_io(graph_def):
    feeds, fetches = utility_library.summarize_io(graph_def)
    return feeds, fetches

def GT(graph_def, input_nodes, output_nodes, transforms):
    output_def = transform_graph(graph_def, input_nodes, output_nodes, transforms)
    return output_def

def CS(graph_def, input_node, input_shape_string):
    output_def = utility_library.process_shapes(graph_def, input_nodes, input_shape_string)
    return output_def

def DF(graph_def):
    output_def = utility_library.format_data(graph_def)
    return output_def

parser = argparse.ArgumentParser(description='TensorFlow Graph Surgery Tool. Performs various preprocessing and optimization steps.')
parser.add_argument('-p', '--input-graph', type=str, required=True, help='Required argument: input_model.pb')
parser.add_argument('-o', '--output-graph', type=str, default='preprocessed_model.pb', help='Output protobuf')
parser.add_argument('-is', '--input-shape', type=str, default='1,224,224,3', help='4-D shape of input tensor (comma-separated), e.g. 1,224,224,3')
parser.add_argument('-in', '--input-nodes', default=None, help='Name of the input node')
parser.add_argument('-on', '--output-nodes', default=None, help='Name of the output node')
parser.add_argument('-T', '--transforms', type=str, default=DEFAULT_TRANSFORMS, help='Uses default transforms')
parser.add_argument('-f', '--format-data', action='store_true', help='Converts data format from NHWC to NCHW')
args = parser.parse_args()

model = args.input_graph
dest = args.output_graph
input_nodes = args.input_nodes
output_nodes = args.output_nodes
transforms = args.transforms
input_shape_string = args.input_shape

with tf.gfile.GFile(model, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# User did not specify input and output_nodes
if input_nodes is None and output_nodes is None:
    feeds, fetches = summarize_io(graph_def) 
    # Format input strings
    try:
        input_nodes = feeds[0]
    except:
        print('Could not find valid feeds, please specify input nodes using -in FLAG')
        sys.exit(1)
    if len(feeds) > 1:
        for name in feeds[1:]:
            input_nodes = input_nodes + ',' + name
    # Format output strings
    try:
        output_nodes = fetches[0]
    except:
        print('Could not find valid fetches, please specify output nodes using -on FLAG')
        sys.exit(1)
    if len(fetches) > 1:
        for name in fetches[1:]:
            output_nodes = output_nodes + ',' + name
    print('\nNo i/o nodes specified, searching for possible nodes...')
    print('    Input nodes: ', input_nodes)
    print('    Output nodes: ', output_nodes)
    print('    Image dimensions: ', input_shape_string)
else:
    print('\nUser specified i/o nodes detected...')
    print('    Input nodes: ', input_nodes)
    print('    Output nodes: ', output_nodes)
    print('    Image dimensions: ', input_shape_string)

print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Preprocessing Step 1: Graph Transform with default transforms')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
output_def_1 = GT(graph_def, input_nodes, output_nodes, transforms)

print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Preprocessing Step 2: Constantify Shapes, remove redundant Reshapes, set input shapes')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
output_def_2 = CS(output_def_1, input_nodes, input_shape_string)

print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Preprocessing Step 3: Graph Transform to clean Step 2')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
output_def_3 = GT(output_def_2, input_nodes, output_nodes, POST_CS_TRANSFORMS)

if args.format_data is not None:
    print('\n+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Preprocessing Step 4: Convert Data Format to NCHW')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++\n')
    output_def_4 = DF(output_def_3)
    tf.train.write_graph(output_def_4, '.', dest, as_text=False)
else:
    tf.train.write_graph(output_def_3, '.', dest, as_text=False)
