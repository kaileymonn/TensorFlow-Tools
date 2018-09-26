import argparse
import code
import copy
import struct
import sys

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import tensor_shape, tensor_util

# Transforms that suck: merge_duplicate_nodes, remove_attribute(attribute_name=_class) APPLY AT VERY END

def format_axes(i, input_rank):
    NCHW_map = {}
    if input_rank == 4:
        NCHW_map = {0:0, 1:2, 2:3, 3:1, 
                    -4:0, -3:2, -2:3, -1:1}
    elif input_rank == 3:
        NCHW_map = {0:0, 1:2, 2:1,
                    -3:0, -2:2, -1:1}
    elif input_rank == 2:
        NCHW_map = {0:0, 1:1,
                    -2:0, -1:1}
    return NCHW_map[i]

def node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]

def create_pre_transpose_node(input_node, perm):
    nn = node_def_pb2.NodeDef()
    const = node_def_pb2.NodeDef()
    nn.name = input_node.name + '/pre_transpose'
    nn.op = 'Transpose'
    const.name = nn.name + '/perm_const_pre'
    const.op = 'Const'
    nn.attr['T'].CopyFrom(input_node.attr['T'])
    nn.attr['Tperm'].type = 3
    const.attr['dtype'].type = 3
    perm_len = len(perm)
    const.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(perm, shape=[perm_len])))
    nn.input.extend([input_node.input[0]])
    nn.input.extend([const.name])
    return nn, const

def create_post_transpose_node(input_node, perm):
    nn = node_def_pb2.NodeDef()
    const = node_def_pb2.NodeDef()
    nn.name = input_node.name
    nn.op = 'Transpose'
    const.name = nn.name + '/perm_const_post'
    const.op = 'Const'
    nn.attr['T'].CopyFrom(input_node.attr['T'])
    nn.attr['Tperm'].type = 3
    const.attr['dtype'].type = 3
    perm_len = len(perm)
    const.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(perm, shape=[perm_len])))
    nn.input.extend([input_node.name + '/middle_of_transposes'])
    nn.input.extend([const.name])
    return nn, const

def create_expand_dim_node(name, input_name, dtype, val):
    nn = node_def_pb2.NodeDef()
    const = node_def_pb2.NodeDef()
    nn.name = name
    nn.op = 'ExpandDims'
    const.name = nn.name + '/const_dim'
    const.op = 'Const'
    nn.attr['T'].type = dtype.as_datatype_enum
    const.attr['dtype'].type = 3
    const.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(val, shape=[])))
    nn.input.extend([input_name])
    nn.input.extend([const.name])
    return nn, const

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

def evaluate_new_placeholders(graph_def, input_nodes, feeds, feed_shapes):
    """
    Called when user passes -in flag. Tries to determine input shapes automatically for user.
    """
    name_to_output = {}
    input_nodes_local = input_nodes.copy()
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    # Generate feed_tensors and feed_dict
    feed_tensors = [graph.get_operation_by_name(feed).outputs[0] for feed in feeds]
    feed_dict = {}
    for i,shape in enumerate(feed_shapes):
        shape_array = [int(x.strip()) for x in shape.split(',')]
        feed_dict[feed_tensors[i]] = np.random.rand(*tuple(shape_array))
    with tf.Session(graph=graph) as sess:
        while len(input_nodes_local) > 0:
            name = input_nodes_local.pop(0)
            try:
                output_tensor = sess.graph.get_operation_by_name(name).outputs[0] 
                output = sess.run(output_tensor, feed_dict=feed_dict)
                name_to_output[name] = output
            except:
                print('Could not determine shape for node {}, please enter input shape manually'.format(name))
                sys.exit(1)
    return name_to_output

def process_shapes(graph_def, feeds, feed_shapes):
    """
    Called when user passes -cs flag. Tries to remove as many dynamic shape inferences as possible.
    Unexpected behavior if run directly after -gt that generates a new subgraph with new feeds. 
    In that case, user should run -cs in a fresh execution with the new subgraph as the input graph, 
    specifying new feed_shapes if required.
    """
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    # Store Ops which we will want to check for output shapes later
    outputs = [n.outputs[0] for n in graph.get_operations() if n.type == 'Shape']
    for n in graph.get_operations():
        if n.type == 'Shape':
            outputs.append(n.outputs[0])
        elif n.type == 'Reshape' and n.inputs[1].op.type != 'Const':
            outputs.append(n.inputs[1].op.outputs[0])

    # Generate feed_tensors and feed_dict
    feed_tensors = [graph.get_operation_by_name(feed).outputs[0] for feed in feeds]
    feed_dict = {}
    for i,shape in enumerate(feed_shapes):
        shape_array = [int(x.strip()) for x in shape.split(',')]
        feed_dict[feed_tensors[i]] = np.random.rand(*tuple(shape_array))

    with tf.Session(graph=graph) as sess:
        rawOut = sess.run(outputs, feed_dict)

    # Optimize performance. Maps op name to np.array(op output shape)
    name_to_output_shape = {}
    for i,op in enumerate(outputs):
        name_to_output_shape[node_name(op.name)] = rawOut[i]

    new_model = graph_pb2.GraphDef()
    with tf.Session(graph=graph) as sess:
        for n in sess.graph_def.node:
            if n.name in feeds:
                nn = new_model.node.add()
                nn.op = 'Placeholder'
                nn.name = n.name
                nn.attr['dtype'].CopyFrom(n.attr['dtype'])
                shape = [int(x.strip()) for x in feed_shapes[feeds.index(n.name)].split(',')]
                nn.attr["shape"].CopyFrom(attr_value_pb2.AttrValue(shape=tensor_shape.TensorShape(shape).as_proto()))

            elif n.op == 'Shape':
                nn = new_model.node.add()
                nn.name = n.name
                nn.op = 'Const'
                nn.attr['dtype'].type = 3
                nn.attr['value'].tensor.dtype = 3
                shape = name_to_output_shape[nn.name]
                temp = nn.attr['value'].tensor.tensor_shape.dim.add()
                temp.size = shape.size
                nn.attr['value'].tensor.tensor_content = np.array(shape).astype(np.int32).tobytes()
            
            # Handle redundant Reshape ops
            elif n.op == 'Reshape':
                nn = new_model.node.add()
                nn.CopyFrom(n)
                # Get graph operation and compare io shapes
                op = sess.graph.get_operation_by_name(n.name)
                # Reshape is redundant
                if(op.inputs[0].shape and op.inputs[0].shape.as_list() == op.outputs[0].shape.as_list()
                                     and op.inputs[1].op.type == 'Const'):
                    temp = node_def_pb2.NodeDef()
                    temp.op = 'Identity'
                    temp.name = nn.name
                    temp.input.extend([nn.input[0]])
                    temp.attr['T'].CopyFrom(nn.attr['T'])
                    new_model.node.pop()
                    new_model.node.extend([temp])

                # Handle case where Shape input is not a const
                if op.inputs[1].op.type != 'Const':
                    input1_name = nn.input[1]
                    shape_content = name_to_output_shape[input1_name]
                    data = np.reshape(shape_content, (np.product(shape_content.shape)))

                    # Create new Const Shape input and set to original input1 name
                    shape_node = new_model.node.add()
                    shape_node.op = 'Const'
                    shape_node.name = nn.name + '/shape_const'
                    shape_node.attr['dtype'].CopyFrom(n.attr['Tshape'])
                    shape_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(data, shape=data.shape)))
                    nn.input.pop()
                    nn.input.extend([shape_node.name])

            elif n.op == 'Conv2D':
                nn = new_model.node.add()
                nn.CopyFrom(n)
                op = sess.graph.get_operation_by_name(n.name)
                if op.inputs[0].op.type == 'Pad':
                    data_format = n.attr['data_format'].s.decode('utf-8')
                    if data_format == 'NHWC':
                        h_index = 1
                        w_index = 2
                    else:
                        h_index = 0
                        w_index = 1
                    pad_type = n.attr['padding'].s.decode('utf-8')
                    in_rank = len(op.inputs[0].shape)
                    pad_bstring = op.inputs[0].op.inputs[1].op.node_def.attr['value'].tensor.tensor_content
                    unpack_format = '<' + 'l'*in_rank*2
                    paddings = struct.unpack(unpack_format, pad_bstring)
                    og_out_shape = list(np.shape(op.outputs[0].eval(feed_dict)))
                    og_in_shape = list(np.shape(op.inputs[0].eval(feed_dict)))
                    kernel_shape = list(np.shape(op.inputs[1].eval(feed_dict)))
                    strides = list(n.attr['strides'].list.i)
                    unpadded_in_shape = [d-paddings[i*2]-paddings[i*2+1] for i,d in enumerate(og_in_shape)]
                    if pad_type == 'VALID':
                        test_height = np.ceil(unpadded_in_shape[h_index]/strides[h_index])
                        test_width = np.ceil(unpadded_in_shape[w_index]/strides[w_index])
                    else:
                        test_height = np.ceil(unpadded_in_shape[h_index]-kernel_shape[0]+1/strides[h_index])
                        test_width = np.ceil(unpadded_in_shape[w_index]-kernel_shape[1]+1/strides[w_index])
                    if int(test_height) == og_out_shape[h_index] and int(test_width) == og_out_shape[w_index]:
                        # Change pad type
                        if pad_type == 'VALID':
                            nn.attr['padding'].s = 'SAME'.encode('utf-8')
                        else:
                            nn.attr['padding'].s = 'VALID'.encode('utf-8')
                        # Disconnect Pad input
                        nn.input[0] = node_name(op.inputs[0].op.inputs[0].name)

            else:
                nn = new_model.node.add()
                nn.CopyFrom(n)

    return new_model

def format_data(graph_def):
    """
    Tries to convert NHWC to NCHW data format.
    Highly recommended to run -df only after having run -cs on the graph.
    Assumes first dimension is always kept as the batch dimension regardless of rank
    """
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    output_graph_def = graph_pb2.GraphDef()
    for n in graph_def.node:
        # Initial copying
        nn = output_graph_def.node.add()
        nn.CopyFrom(n)
        op = graph.get_operation_by_name(n.name)

        # Ops like Convolutions, Pooling, BiasAdd, FusedBatchNorm etc...
        if 'data_format' in nn.attr and len(op.outputs[0].shape) > 2:
            nn.attr['data_format'].s = "NCHW".encode('utf-8')
            # Handle other attributes like strides, dilations
            for attr in nn.attr:
                if nn.attr[attr].list and len(nn.attr[attr].list.i) == 4:
                    data = list(nn.attr[attr].list.i)
                    order = [0,3,1,2]
                    new_data = [data[i] for i in order]
                    for i,dim in enumerate(new_data):
                        nn.attr[attr].list.i[i] = dim
        
        # Placeholder ops
        elif nn.op == 'Placeholder':
            if op.outputs[0].shape:
                output_rank = len(op.outputs[0].shape)
            else:
                continue # Unknown Placeholder input shape, nothing to do i suppose
            if output_rank == 4:
                order = [0,3,1,2]
            elif output_rank == 3:
                order = [0,2,1]
            elif output_rank == 2:
                order = [0,1]
            else:
                continue # Nothing to change for 1D/Scalars
            shape = [x.size for x in list(nn.attr['shape'].shape.dim)]
            new_shape = [shape[i] for i in order]
            for i,dim in enumerate(new_shape):
                nn.attr['shape'].shape.dim[i].size = dim

        # Reshape, Pad ops
        elif nn.op in set(['Reshape', 'Pad']):
            if op.inputs[0].shape:
                input_rank = len(op.inputs[0].shape)
            else:
                if nn.op == 'Pad':
                    input_rank = len(op.outputs[0].shape)
                else:
                    continue
            for node in output_graph_def.node:
                if node.name == nn.input[1]:
                    # Reshape
                    if nn.op == 'Reshape':
                        output_rank = node.attr['value'].tensor.tensor_shape.dim[0].size
                        if output_rank == 4:
                            order = [0,3,1,2]
                        elif output_rank == 3:
                            order = [0,2,1]
                        elif output_rank == 2:
                            order = [0,1]
                        elif output_rank != 0 and output_rank != 1:
                            print('Unsupported output rank for Reshape op: ', nn.name)
                            code.interact(local=locals())
                        pack_format = '<' + 'l'*output_rank
                        bstring = node.attr['value'].tensor.tensor_content
                        if len(bstring) != 0:
                            content = struct.unpack(pack_format, bstring)
                            new_content = [content[i] for i in order]
                            node.attr['value'].tensor.tensor_content = struct.pack(pack_format, *tuple(new_content))
                    # Pad
                    elif nn.op == 'Pad':
                        output_rank = node.attr['value'].tensor.tensor_shape.dim[0].size
                        if output_rank == 4:
                            order = [0,1,6,7,2,3,4,5]
                        elif output_rank == 3:
                            order = [0,1,4,5,2,3]
                        elif output_rank == 2:
                            order = [0,1,2,3]
                        elif output_rank != 0 and output_rank != 1:
                            print('Unsupported output rank for Pad op: ', nn.name)
                            code.interact(local=locals())
                        pack_format = '<' + 'l'*output_rank*2
                        bstring = node.attr['value'].tensor.tensor_content
                        content = struct.unpack(pack_format, bstring)
                        new_content = [content[i] for i in order]
                        node.attr['value'].tensor.tensor_content = struct.pack(pack_format, *tuple(new_content))

        # Ops that modify shape along a single dimension axis (specified as a dimension input node)
        elif nn.op in set(['ConcatV2', 'Split']):
            if nn.op == 'ConcatV2':
                num_inputs = len(op.inputs)
                dim_input_name = nn.input[num_inputs-1]
                try:
                    input_rank = len(op.inputs[0].shape)
                except Exception as e:
                    print(e, '\n Could not determine input shape for {}. Passing...'.format(nn.name))
                    pass
            elif nn.op == 'Split':
                dim_input_name = nn.input[0]
                try:
                    input_rank = len(op.inputs[1].shape)
                except Exception as e:
                    print(e, '\n Could not determine input shape for {}. Passing...'.format(nn.name))
                    pass
            for node in output_graph_def.node:
                if node.name == dim_input_name:
                    axis = node.attr['value'].tensor.int_val[0]
                    if input_rank in [2,3,4]:
                        node.attr['value'].tensor.int_val[0] = format_axes(axis, input_rank)
                    elif input_rank != 0 and input_rank != 1:
                        print('Unsupported input rank for {} op: '.format(nn.op), nn.name)
                        code.interact(local=locals())

        # Ops that modify shape along dimension axes (specified as attribute)
        elif nn.op in set(['Squeeze', 'Unpack', 'Pack']):
            if op.inputs[0].shape:
                input_rank = len(op.inputs[0].shape)
            else:
                continue # Unknown input shape, skip
            if nn.op == 'Squeeze':
                if input_rank in [2,3,4]:
                    for i,axis in enumerate(nn.attr['squeeze_dims'].list.i):
                        nn.attr['squeeze_dims'].list.i[i] = format_axes(axis, input_rank)
            elif nn.op == 'Unpack' or nn.op == 'Pack':
                axis = nn.attr['axis'].i
                if input_rank in [2,3,4]:
                    nn.attr['axis'].i = format_axes(axis, input_rank)

        # Ops that modify shape along multiple dimension axes (specifed as a dimension input node)
        elif nn.op == 'Mean':
            if op.inputs[0].shape:
                input_rank = len(op.inputs[0].shape)
            else:
                continue # Unknown input shape, nothing to do
            for node in output_graph_def.node:
                if node.name == nn.input[1]:            
                    num_axes = node.attr['value'].tensor.tensor_shape.dim[0].size
                    pack_format = '<' + 'l'*num_axes
                    bstring = node.attr['value'].tensor.tensor_content
                    content = struct.unpack(pack_format, bstring)
                    if input_rank in [2,3,4]:
                        new_content = [format_axes(i, input_rank) for i in content]
                        node.attr['value'].tensor.tensor_content = struct.pack(pack_format, *tuple(new_content))
                    elif input_rank != 0 and input_rank != 1:
                        print('Unsupported input rank for {} op: '.format(nn.op), nn.name)
                        code.interact(local=locals())

        # Ops that require NHWC inputs and generate NHWC outputs, bound with transpose ops
        elif nn.op in set(['CropAndResize', 'SpaceToBatchND', 'BatchToSpaceND']):
            perm_out = [0,3,1,2]   
            perm_in = [0,2,3,1]
            tn1, tc1 = create_pre_transpose_node(nn, perm_in)
            tn2, tc2 = create_post_transpose_node(nn, perm_out)
            assert(tn2.name == nn.name)
            assert(tn1.input[0] == nn.input[0])
            nn.name = nn.name + '/middle_of_transposes'
            nn.input[0] = tn1.name
            output_graph_def.node.extend([tc1,tc2,tn1,tn2])

        # Arithmetic ops that that require implicit broadcasting
        elif nn.op in set(['Add', 'Sub', 'Mul', 'ReadDiv', 'Minimum', 'Maximum']):
            if(op.inputs[0].shape and op.inputs[1].shape and len(op.inputs[0].get_shape()) != len(op.inputs[1].get_shape())):
                main_index = 0 if len(op.inputs[0].get_shape()) > len(op.inputs[1].get_shape()) else 1
                other_index = 1 if main_index == 0 else 0
                main_shape = op.inputs[main_index].shape
                other_shape = op.inputs[other_index].shape
                # Don't have to handle broadcasting of scalar to N-D tensor
                if len(other_shape) > 0:
                    window_len = len(other_shape)
                    start_index = 0
                    while start_index + window_len < len(main_shape):
                        if main_shape.as_list()[start_index:start_index+window_len] == other_shape.as_list():
                            break
                        else:
                            start_index += 1
                    if start_index == 3: # Bcast 1D to 4D along channels dim
                        en1, ec1 = create_expand_dim_node(nn.name+'/expand_dim_1', nn.input[other_index], op.inputs[other_index].dtype, -1)
                        en2, ec2 = create_expand_dim_node(nn.name+'/expand_dim_2', nn.name+'/expand_dim_1', op.inputs[other_index].dtype, -1)
                        nn.input[other_index] = en2.name
                        output_graph_def.node.extend([ec1,ec2,en1,en2])

        # Strided Slice Op is such a pain...
        elif nn.op == 'StridedSlice':
            if op.inputs[0].shape:
                input_rank = len(op.inputs[0].shape)
            else:
                continue
            begin_name = nn.input[1]
            end_name = nn.input[2]
            strides_name = nn.input[3]
            # Handle inputs
            for node in output_graph_def.node:
                if node.name in [begin_name, end_name, strides_name]:
                    if input_rank == 4:
                        order = [0,3,1,2]
                    elif input_rank == 3:
                        order = [0,2,1]
                    elif input_rank == 2:
                        order = [0,1]
                    elif input_rank != 0 and input_rank != 1:
                        print('Unsupported input rank for StridedSlice op: ', nn.name)
                        code.interact(local=locals())                    
                    bstring = node.attr['value'].tensor.tensor_content
                    if len(bstring) == 0:
                        content = node.attr['value'].tensor.int_val[0]
                        if input_rank > 1:
                            node.attr['value'].tensor.int_val[0] = format_axes(content, input_rank)
                    else:
                        pack_format = '<'+'l'*input_rank  
                        content = struct.unpack(pack_format, bstring)
                        new_content = [content[i] for i in order]
                        node.attr['value'].tensor.tensor_content = struct.pack(pack_format, *tuple(new_content))
            # Handle mask attributes
            for attr in nn.attr:
                if attr in set(['begin_mask', 'ellipsis_mask', 'end_mask', 'new_axis_mask', 'shrink_axis_mask']):
                    if nn.attr[attr].i != 0 and input_rank > 1:
                        val = nn.attr[attr].i
                        indices = []
                        while val > 1:
                            bit = int(np.log2(val))
                            indices.append(bit)
                            val = val - np.power(2,bit)
                        if val == 1:
                            indices.append(0)
                        lil_endian_indices = [input_rank-1-i for i in indices]
                        print(indices)
                        print(lil_endian_indices)
                        print(nn.attr[attr].i)
                        temp_indices = [format_axes(i, input_rank) for i in lil_endian_indices]
                        indices = [input_rank-1-i for i in temp_indices]
                        nn.attr[attr].i = np.sum([np.power(2, i) for i in indices])
    assert(len(output_graph_def.node) >= len(graph_def.node))
    return output_graph_def
