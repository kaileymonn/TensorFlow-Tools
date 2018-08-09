#!/usr/bin/env python3

import argparse
import copy
import sys
import six
import os

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import tensor_util


def node_name(n):
	if n.startswith("^"):
		return n[1:]
	else:
		return n.split(":")[0]

def extract_graph_summary(graph_def):
	name_to_input_name = {}  # Keyed by the dest node name.
	name_to_node = {}  # Keyed by node name.

	# Keeps track of node sequences. It is important to still output the
	# operations in the original order.
	name_to_seq_num = {}  # Keyed by node name.
	seq = 0
	for node in graph_def.node:
		n = node_name(node.name)
		name_to_node[n] = node
		name_to_input_name[n] = [node_name(x) for x in node.input]
		name_to_seq_num[n] = seq
		seq += 1
	return name_to_input_name, name_to_node, name_to_seq_num

def assert_nodes_are_present(name_to_node, nodes):
    for d in nodes:
        assert d in name_to_node, "%s is not in graph" % d

def bfs_for_reachable_nodes(target_nodes, name_to_input_name):
	nodes_to_keep = set()
	# Breadth first search to find all the nodes that we should keep.
	next_to_visit = target_nodes[:]
	while next_to_visit:
		n = next_to_visit[0]
		del next_to_visit[0]
		if n in nodes_to_keep:
			# Already visited this node.
			continue
		nodes_to_keep.add(n)
		next_to_visit += name_to_input_name[n]
	return nodes_to_keep


def extract_subgraph(graph_def, dest_nodes, start_nodes):
    if not isinstance(graph_def, graph_pb2.GraphDef):
        raise TypeError("graph_def must be a graph_pb2.GraphDef proto.")

    if isinstance(dest_nodes, six.string_types):
        raise TypeError("dest_nodes must be a list.")

    if isinstance(start_nodes, six.string_types):
        raise TypeError("start_nodes must be a list.")

    name_to_input_name, name_to_node, name_to_seq_num = extract_graph_summary(graph_def)
    if start_nodes is not None:
        assert_nodes_are_present(name_to_node, dest_nodes)
        assert_nodes_are_present(name_to_node, start_nodes)
    else:
        assert_nodes_are_present(name_to_node, dest_nodes)

    # Unspecified start_nodes, just cut graph at output_nodes and get downstream nodes
    if start_nodes is None:
        nodes_to_keep = bfs_for_reachable_nodes(dest_nodes, name_to_input_name)
        nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: name_to_seq_num[n])
        # Now construct the output GraphDef
        out = graph_pb2.GraphDef()
        for n in nodes_to_keep_list:
            out.node.extend([copy.deepcopy(name_to_node[n])])
        out.library.CopyFrom(graph_def.library)
        out.versions.CopyFrom(graph_def.versions)

        return out

    # Perform mid-cut
    else:
        endpoints = []

        nodes_superset = bfs_for_reachable_nodes(dest_nodes, name_to_input_name)
        nodes_to_dump = bfs_for_reachable_nodes(start_nodes, name_to_input_name)
        
        nodes_superset_list = sorted(list(nodes_superset), key=lambda n: name_to_seq_num[n])
        nodes_to_dump_list = sorted(list(nodes_to_dump), key=lambda n: name_to_seq_num[n])

        nodes_to_keep = list(set(nodes_superset_list) - set(nodes_to_dump_list))
        nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: name_to_seq_num[n])

        # Now construct the output GraphDef
        out = graph_pb2.GraphDef()
        # temp_list = []
        for n in nodes_to_keep_list:               
            for raw_input in name_to_node[n].input:
                input_name = node_name(raw_input)
                # TODO: Janky way of dealing with cropandreisze, please fix
                if name_to_node[input_name].op == 'CropAndResize':
                    new_node = node_def_pb2.NodeDef()
                    new_placeholder = node_def_pb2.NodeDef()
                    new_const = node_def_pb2.NodeDef()
                    new_const = copy.deepcopy(name_to_node['CropAndResize/crop_size'])
                    new_node.name = input_name
                    new_placeholder.name = 'Placeholder'
                    new_node.op = 'ResizeBilinear'
                    new_node.input.append('Placeholder')
                    new_node.input.append(new_const.name)
                    new_placeholder.op = 'Placeholder'
                    new_node.attr['T'].CopyFrom(name_to_node[n].attr['T'])
                    new_placeholder.attr['dtype'].CopyFrom(name_to_node[n].attr['T'])
                    endpoints.append(new_placeholder.name)
                    out.node.extend([new_node])
                    out.node.extend([new_const])
                    out.node.extend([new_placeholder])
                    continue            
                if input_name not in nodes_to_keep_list and input_name not in endpoints:
                    new_node = node_def_pb2.NodeDef()
                    # # TODO: More janky stuff
                    # if input_name == 'SecondStagePostprocessor/strided_slice_1' or input_name == 'SecondStagePostprocessor/strided_slice_2':
                    #     if input_name not in temp_list:
                    #         new_node = copy.deepcopy(name_to_node[input_name])
                    #         new_node.input[0] = 'Shape'
                    #         for const in new_node.input:
                    #             if const != 'Shape' and const not in temp_list:
                    #                 new_const = node_def_pb2.NodeDef()
                    #                 new_const = copy.deepcopy(name_to_node[const])
                    #                 temp_list.append(new_const.name)
                    #                 out.node.extend([new_const])
                    #         temp_list.append(input_name)
                    #         out.node.extend([new_node])
                    #         continue
                    #     else:
                    #         continue
                    new_node.name = input_name
                    new_node.op = 'Placeholder'
                    if 'dtype' in name_to_node[n].attr:
                        new_node.attr['dtype'].CopyFrom(name_to_node[n].attr['dtype'])
                    elif 'T' in name_to_node[n].attr:
                        new_node.attr['dtype'].CopyFrom(name_to_node[n].attr['T'])
                    else:
                        print('Node {} is not a great choice for a start node because of its DType, please try again.')
                        import code
                        code.interact(local=locals())

                    endpoints.append(new_node.name)
                    out.node.extend([new_node])

                # # Clean up 'Enter' nodes
                # if name_to_node[n].op == 'Enter':
                #     # import code
                #     # code.interact(local=locals())
                #     for item in name_to_node[n].input:
                #         name_to_node[n].input.remove(item)
                #         name_to_node[n].op = 'Placeholder'
                #         dtype = name_to_node[n].attr['T'].type
                #         name_to_node[n].attr.clear()
                #         name_to_node[n].attr['dtype'].type = dtype
            out.node.extend([copy.deepcopy(name_to_node[n])])
        out.library.CopyFrom(graph_def.library)
        out.versions.CopyFrom(graph_def.versions)

        print('Feeds: ', endpoints)
        print('Fetches: ', dest_nodes)
        return out

def convert_variables_to_constants(sess,
                                   input_graph_def,
                                   output_node_names,
                                   start_nodes=None,
                                   variable_names_whitelist=None,
                                   variable_names_blacklist=None):
    """Replaces all the variables in a graph with constants of the same values.

    If you have a trained graph containing Variable ops, it can be convenient to
    convert them all to Const ops holding the same values. This makes it possible
    to describe the network fully with a single GraphDef file, and allows the
    removal of a lot of ops related to loading and saving the variables.

    Args:
        sess: Active TensorFlow session containing the variables.
        input_graph_def: GraphDef object holding the network.
        output_node_names: List of name strings for the result nodes of the graph.
        input_node_names: List of name strings for the new placeholders of the graph.
        variable_names_whitelist: The set of variable names to convert (by default,
                                all variables are converted).
        variable_names_blacklist: The set of variable names to omit converting
                                to constants.

    Returns:
        GraphDef containing a simplified version of the original.
    """
    # This graph only includes the nodes needed to evaluate the output nodes, and
    # removes unneeded nodes like those involved in saving and assignment.
    inference_graph = extract_subgraph(input_graph_def, output_node_names, start_nodes)

    found_variables = {}
    variable_names = []
    variable_dict_names = []
    for node in inference_graph.node:
        if node.op in ["Variable", "VariableV2", "VarHandleOp"]:
            variable_name = node.name
            if ((variable_names_whitelist is not None and
                variable_name not in variable_names_whitelist) or
                (variable_names_blacklist is not None and
                variable_name in variable_names_blacklist)):
                continue
            variable_dict_names.append(variable_name)
            if node.op == "VarHandleOp":
                variable_names.append(variable_name + "/Read/ReadVariableOp:0")
            else:
                variable_names.append(variable_name + ":0")
    if variable_names:
        returned_variables = sess.run(variable_names)
    else:
        returned_variables = []
    found_variables = dict(zip(variable_dict_names, returned_variables))
    logging.info("Froze %d variables.", len(returned_variables))

    output_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = node_def_pb2.NodeDef()
        if input_node.name in found_variables:
            output_node.op = "Const"
            output_node.name = input_node.name
            dtype = input_node.attr["dtype"]
            data = found_variables[input_node.name]
            output_node.attr["dtype"].CopyFrom(dtype)
            output_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(
                        data, dtype=dtype.type, shape=data.shape)))
            how_many_converted += 1
        elif input_node.op == "ReadVariableOp" and (input_node.input[0] in found_variables):
            # The preceding branch converts all VarHandleOps of ResourceVariables to
            # constants, so we need to convert the associated ReadVariableOps to
            # Identity ops.
            output_node.op = "Identity"
            output_node.name = input_node.name
            output_node.input.extend([input_node.input[0]])
            output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
            if "_class" in input_node.attr:
                output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
        else:
            output_node.CopyFrom(input_node)
        output_graph_def.node.extend([output_node])

    output_graph_def.library.CopyFrom(inference_graph.library)
    logging.info("Converted %d variables to const ops.", how_many_converted)
    return output_graph_def

#---------------------------------------------------------------------------
# Parse commandline inputs
#---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Exports a tensorflow model as a .pb file')
parser.add_argument('--metagraph-file', default='saved_bin/final.ckpt.meta', help='name of the metagraph file')
parser.add_argument('--checkpoint-file', default='saved_bin/final.ckpt', help='name of the checkpoint file')
parser.add_argument('--output-file', default='model.pb', help='name of the output file')
parser.add_argument('--output-tensors', nargs='+', required=True, help='names of the output tensors')
parser.add_argument('--input-tensors', nargs='+', default=None, help='names of the input tensors')
args = parser.parse_args()

print('[i] Matagraph file:  ', args.metagraph_file)
print('[i] Checkpoint file: ', args.checkpoint_file)
print('[i] Output file:     ', args.output_file)
print('[i] Output tensors:  ', args.output_tensors)
print('[i] Input tensors:  ', args.input_tensors)

for f in [args.checkpoint_file+'.index', args.metagraph_file]:
    # import code
    # code.interact(local=locals())
    if not os.path.exists(f):
        print('[!] Cannot find file:', f)
        sys.exit(1)
#-------------------------------------------------------------------------------
# Export the graph
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(args.metagraph_file)
    saver.restore(sess, args.checkpoint_file)

    graph = tf.get_default_graph()
    # outputs = [n.name for n in graph.get_operations() if len(n.outputs) == 0]
    # print(outputs)
    input_graph_def = graph.as_graph_def()
    output_graph_def = convert_variables_to_constants(sess, input_graph_def, args.output_tensors, args.input_tensors)

    with open(args.output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())


