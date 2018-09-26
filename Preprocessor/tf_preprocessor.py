#!/usr/bin/env python3

import argparse
import code
import sys

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import tensor_shape

import utility_library
from graph_transform import TransformGraph as transform_graph

BASE_TRANSFORMS = 'strip_unused_nodes flatten_atrous_conv fold_constants fold_batch_norms fold_old_batch_norms remove_nodes(op=Identity, op=CheckNumerics, op=Assert) remove_attribute(attribute_name=_class) strip_unused_nodes sort_by_execution_order'

np_dtype_map = {
    np.dtype('float32'): 'float',
    np.dtype('float64'): 'float',
    np.dtype('int8'): 'int8',
    np.dtype('uint8'): 'uint8',
    np.dtype('int16'): 'int16',
    np.dtype('uint16'): 'uint16',
    np.dtype('int32'): 'int32',
    np.dtype('int64'): 'int64'
}

def summarize_io(graph_def):
    feeds, fetches = utility_library.summarize_io(graph_def)
    return feeds, fetches

def get_feed_shapes(graph_def, feeds):
    output = [] # List of shapes as strings
    feed_shapes_are_static = True
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    for feed in feeds:
        feed_shape = graph.get_operation_by_name(feed).outputs[0].shape.as_list()
        for i in feed_shape:
            if i is None:
                feed_shapes_are_static = False
        temp = ['{}'.format(i) for i in feed_shape]
        output.append(','.join(temp))
    return output, feed_shapes_are_static

def evaluate_new_placeholders(graph_def, input_nodes, feeds, feed_shapes):
    name_to_output = utility_library.evaluate_new_placeholders(graph_def, input_nodes, feeds, feed_shapes)
    return name_to_output

def GT(graph_def, input_nodes, output_nodes, transforms):
    # Format argument strings for transform_graph()
    input_nodes_string = ','.join(input_nodes)
    output_nodes_string = ','.join(output_nodes)
    input_nodes_string = input_nodes_string.encode('utf-8')
    output_nodes_string = output_nodes_string.encode('utf-8')
    output_def = transform_graph(graph_def, input_nodes_string, output_nodes_string, transforms)
    return output_def

def CS(graph_def, feeds, feed_shapes):
    output_def = utility_library.process_shapes(graph_def, feeds, feed_shapes)
    return output_def

def DF(graph_def):
    output_def = utility_library.format_data(graph_def)
    return output_def

parser = argparse.ArgumentParser(description='TensorFlow Graph Surgery Tool. Performs various preprocessing and optimization steps.')
parser.add_argument('-p', '--input-graph', type=str, required=True, help='Required argument: input_model.pb')
parser.add_argument('-o', '--output-graph', type=str, default='optimized_model.pb', help='Output protobuf')
parser.add_argument('-fs', '--feed-shapes', default=['1,224,224,3'], nargs='+', help='Shapes of feed tensors (Comma-and-space-separated), e.g. "1,224,224,3" "1,512,256"')
parser.add_argument('-is', '--input-shapes', default=None, nargs='+', help='Shapes of input tensors (Comma-and-space-separated), e.g. "1,224,224,3" "1,512,256"')
parser.add_argument('-in', '--input-nodes', default=None, nargs='+', help='Name of the input nodes (Comma-and-space-separated) e.g. "Placeholder1" "Placeholder2"')
parser.add_argument('-on', '--output-nodes', default=None, nargs='+', help='Name of the output nodes (Comma-and-space-separated) e.g. "Fetch1" "Fetch2"')
parser.add_argument('-cs', '--constantify-shapes', action='store_true', help='Some shape magic')
parser.add_argument('-gt', '--transforms', nargs='?', const="{}".format(BASE_TRANSFORMS), default=None, help='Apply Graph Transform transforms')
parser.add_argument('-df', '--format-data', action='store_true', help='Converts data format from NHWC to NCHW')
parser.add_argument('-all', '--all', action='store_true', help='Runs -cs, -gt and -df in this order')
args = parser.parse_args()

model = args.input_graph
dest = args.output_graph
feed_shapes = args.feed_shapes
input_shapes = args.input_shapes
input_nodes = args.input_nodes
output_nodes = args.output_nodes

with tf.gfile.GFile(model, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Get feed and fetch names of the graph_def
feeds, fetches = summarize_io(graph_def)

# Check if # feed_shapes match with # feeds, # feed_shapes = 1 by default
if len(feed_shapes) != len(feeds):
    feed_shapes, are_static = get_feed_shapes(graph_def, feeds)
    if are_static == False:
        print('Error: Mismatching number of feed nodes and feed shapes, could not find static feed shapes.')
        print('There were {0} feeds detected but {1} feed shapes specified.'.format(len(feeds), len(feed_shapes)))
        for feed in feeds:
            print('     Feed: {}'.format(feed))
        print('Please specify feed shapes for these feeds using the -fs flag.')
        sys.exit(1)
# Check if ranks match
else:
    temp, are_static = get_feed_shapes(graph_def, feeds)
    for i,shape_string in enumerate(feed_shapes):
        if len(shape_string.split(',')) != len(temp[i].split(',')):
            print('Error: Invalid rank of feed shape for feed {}'.format(feeds[i]))
            print('Required rank: {}'.format(len(temp[i].split(','))))
            print('Received bad rank of {}'.format(len(shape_string.split(','))))
            for j,feed in enumerate(feeds):
                print('     Feed: {0}, rank: {1}'.format(feed, len(temp[j].split(','))))
            print('Please specify feed shapes for these feeds using the -fs flag.')
            sys.exit(1)

if output_nodes is None:
    output_nodes = fetches.copy()
    if len(fetches) == 0:
        print('Could not find valid fetches, please specify output nodes using -on FLAG')
        sys.exit(1)

if input_nodes is None:
    input_nodes = feeds.copy()
    input_shapes = feed_shapes.copy()
    transforms = BASE_TRANSFORMS
    if len(feeds) == 0:
        print('Could not find valid feeds, please specify input nodes using -in FLAG')
        sys.exit(1)
else:
    name_to_output = evaluate_new_placeholders(graph_def, input_nodes, feeds, feed_shapes)
    if input_shapes is not None:
        if len(input_shapes) != len(input_nodes):
            print('Error: Mismatching number of input nodes and input shapes.')
            print('There were {0} input nodes paired with {1} input shapes'.format(len(input_nodes), len(input_shapes)))
            sys.exit(1)
        # Check rank of input shapes
        temp = []
        for key, val in name_to_output.items():
            temp.append(','.join(['{}'.format(i) for i in np.shape(val)]))
        for i,shape_string in enumerate(input_shapes):
            if len(shape_string.split(',')) != len(temp[i].split(',')):
                print('Error: Invalid rank of input shape for input node {}'.format(input_nodes[i]))
                print('Required rank: {}'.format(len(temp[i].split(','))))
                print('Received bad rank of {}'.format(len(shape_string.split(','))))
                for j,name in enumerate(input_nodes):
                    print('     input: {0}, rank: {1}'.format(name, len(temp[j].split(','))))
                print('Please specify input shapes for these inputs using the -is flag.')
                sys.exit(1)
    else:
        input_shapes = []
        for key, val in name_to_output.items():
            temp = ['{}'.format(i) for i in np.shape(val)] 
            input_shapes.append(','.join(temp))

    # Generate new strip_unused_nodes arg string
    new_string = 'strip_unused_nodes('
    strip_arguments = []
    for i,node in enumerate(input_nodes):
        dtype = np_dtype_map[name_to_output[node].dtype]
        shape_string = input_shapes[i]
        new_placeholder = ['name={0}'.format(node), 'type_for_name={0}'.format(dtype), 'shape_for_name=\"{0}\"'.format(shape_string)]
        strip_arguments.append(', '.join(new_placeholder))
    new_string = new_string + ', '.join(strip_arguments) + ')'
    transforms = BASE_TRANSFORMS.split(' ')
    transforms[0] = new_string
    transforms = ' '.join(transforms)

print('\n    Feed nodes: ', feeds)
print('    Feed shapes: ', feed_shapes)    
print('    Input nodes: ', input_nodes)
print('    Input shapes: ', input_shapes)
print('    Output nodes: ', output_nodes)

last_output_def = graph_def

# Parse optimization arguments in order
for i,arg in enumerate(sys.argv):
    if arg[0] == '-':
        if arg == '-all':
            print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Preprocessing Step 1: Removing dynamic shape inferencing')  
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            output_def_1 = CS(last_output_def, feeds, feed_shapes)
            last_output_def = output_def_1

            print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Preprocessing Step 2: Graph Transform with default arguments')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            output_def_2 = GT(output_def_1, input_nodes, output_nodes, transforms)
            last_output_def = output_def_2

            print('\n++++++++++++++++++++++++++++++++++++++ ++++++++++')
            print('Preprocessing Step 3: Convert Data Format to NCHW')
            print('+++++++++++++++++++++++++++++++++++ +++++++++++++\n')
            print('Warning: This is still a WIP\n')
            output_def_3 = DF(last_output_def)
            last_output_def = output_def_3

        elif arg == '-gt':
            print('\n+++++++++++++++++++++++++++++++++++++++++++')
            print('Single-Step Preprocessing: Graph Transforms')
            print('+++++++++++++++++++++++++++++++++++++++++++\n')
            if i < len(sys.argv) - 1 and sys.argv[i+1][0] != '-':
                transforms = sys.argv[i+1].strip('\'')
            else:
                transforms = BASE_TRANSFORMS
            print('Transforms: {}\n'.format(transforms))       
            output_def = GT(last_output_def, input_nodes, output_nodes, transforms)
            last_output_def = output_def

        elif arg == '-cs':
            print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Single-Step Preprocessing: Removing dynamic shape inferencing')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            output_def = CS(last_output_def, feeds, feed_shapes)
            last_output_def = output_def

        elif arg == '-df':
            print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Single-Step Preprocessing: Converting NHWC to NCHW')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            print('Warning: Performing this without first applying -cs may not yield desired results\n')
            output_def = DF(last_output_def)
            last_output_def = output_def

tf.train.write_graph(last_output_def, '.', dest, as_text=False)
