import argparse
import code
import sys
import struct

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2, attr_value_pb2, node_def_pb2
from tensorflow.python.framework import tensor_shape

def format_axes(i):
    NCHW_map = {0:0, 1:2, 2:3, 3:1}
    return NCHW_map[i]

def node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]

def summarize_io(graph_def):
    feeds = set()
    fetches = set()
    visited = set()
    visited_inputs = set()
    
    for n in graph_def.node:
        if n.op == 'Placeholder':
            feeds.add(n.name)
            visited.add(n.name)
        else:
            visited.add(n.name)
            for i in n.input:
                visited_inputs.add(node_name(i))
    fetches = visited - visited_inputs
    return list(feeds), list(fetches)

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
                # Get graph operation and compare io shapes
                op = sess.graph.get_operation_by_name(n.name)
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

def format_data(graph_def):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')    

    new_model = graph_pb2.GraphDef()
    with tf.Session(graph=graph) as sess:
        for n in sess.graph_def.node:
            # For general cases
            if 'data_format' in n.attr:
                nn = new_model.node.add()
                nn.CopyFrom(n)
                nn.attr['data_format'].s = "NCHW".encode('utf-8')
                for attr in nn.attr:
                    if nn.attr[attr].list and len(nn.attr[attr].list.i) == 4:
                        data = list(nn.attr[attr].list.i)
                        order = [0,3,1,2]
                        new_data = [data[i] for i in order]
                        for i,dim in enumerate(new_data):
                            nn.attr[attr].list.i[i] = dim
           
           # For cases like Placeholders
            elif 'shape' in n.attr:
                nn = new_model.node.add()
                nn.CopyFrom(n)
                op = sess.graph.get_operation_by_name(n.name)
                if len(op.outputs[0].shape.as_list()) == 4:
                    shape = [x.size for x in list(nn.attr['shape'].shape.dim)]
                    order = [0,3,1,2]
                    new_shape = [shape[i] for i in order]
                    for i,dim in enumerate(new_shape):
                        nn.attr['shape'].shape.dim[i].size = dim

            # For cases like Reshapes
            elif(n.op == 'Const' and len(n.attr['value'].tensor.tensor_shape.dim) == 1 
                                and n.attr['value'].tensor.tensor_shape.dim[0].size == 4):
                nn = new_model.node.add()
                nn.CopyFrom(n)
                bstring = nn.attr['value'].tensor.tensor_content
                shape = struct.unpack('<llll', bstring)
                order = [0,3,1,2]
                new_shape = [shape[i] for i in order]
                nn.attr['value'].tensor.tensor_content = struct.pack('<llll', *tuple(new_shape))

            
            elif n.op == 'Squeeze':
                nn = new_model.node.add()
                nn.CopyFrom(n)
                op = sess.graph.get_operation_by_name(n.name)
                if op.inputs[0].shape and len(op.inputs[0].shape.as_list()) == 4:
                    for i,axis in enumerate(nn.attr['squeeze_dims'].list.i):
                        nn.attr['squeeze_dims'].list.i[i] = format_axes(axis)
                
            else:
                nn = new_model.node.add()
                nn.CopyFrom(n)
    
    return new_model