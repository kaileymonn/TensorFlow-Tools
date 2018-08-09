#!/usr/bin/env python3
#
# Copyright (c) 2017-2018 Ambarella, Inc.
#
# This file and its contents ("Software") are protected by intellectual property rights including,
# without limitation, U.S. and/or foreign copyrights.  This Software is also the confidential and
# proprietary information of Ambarella, Inc. and its licensors.  You may not use, reproduce, disclose,
# distribute, modify, or otherwise prepare derivative works of this Software or any portion thereof
# except pursuant to a signed license agreement or nondisclosure agreement with Ambarella, Inc. or
# its authorized affiliates.  In the absence of such an agreement, you agree to promptly notify and
# return this Software to Ambarella, Inc.
#
# THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL AMBARELLA, INC. OR ITS AFFILIATES BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; COMPUTER FAILURE OR MALFUNCTION; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import argparse
import copy
import sys
import six

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2


def node_name(n):
	if n.startswith("^"):
		return n[1:]
	else:
		return n.split(":")[0]

def extract_graph_summary(graph_def):
	"""Extracts useful information from the graph and returns them."""
	name_to_input_name = {}  # Keyed by the dest node name.
	name_to_node = {}  # Keyed by node name.

	# Keeps track of node sequences. It is important to still output the
	# operations in the original order.
	name_to_seq_num = {}  # Keyed by node name.
	exit_nodes = []
	seq = 0
	for node in graph_def.node:
		n = node_name(node.name)
		name_to_node[n] = node
		name_to_input_name[n] = [node_name(x) for x in node.input]
		name_to_seq_num[n] = seq

		# Save exit nodes
		if node.op == 'Exit':
			exit_nodes.append(n)
		seq += 1
	return name_to_input_name, name_to_node, name_to_seq_num, exit_nodes

def bfs_for_reachable_nodes(exit_nodes, name_to_input_name, name_to_node):
	"""Breadth first search from exit nodes to enter nodes."""
	nodes_in_loop = set()
	# Breadth first search to find all the nodes that we should keep.
	next_to_visit = exit_nodes[:]
	while next_to_visit:
		n = next_to_visit[0]
		del next_to_visit[0]
		if n in nodes_in_loop:
			# Already visited this node.
			continue
		if name_to_node[n].op == 'Enter':
			# Dont go past 'Enter' nodes
			nodes_in_loop.add(n)
			continue
		nodes_in_loop.add(n)
		next_to_visit += name_to_input_name[n]
	return nodes_in_loop


def extract_loops(graph_def):
	"""Extract the subgraph that can reach any of the nodes in 'dest_nodes'.

	Args:
		graph_def: A graph_pb2.GraphDef proto.
		dest_nodes: A list of strings specifying the destination node names.
	Returns:
		The GraphDef of the sub-graph.

	Raises:
		TypeError: If 'graph_def' is not a graph_pb2.GraphDef proto.
	"""

	if not isinstance(graph_def, graph_pb2.GraphDef):
		raise TypeError("graph_def must be a graph_pb2.GraphDef proto.")

	name_to_input_name, name_to_node, name_to_seq_num, exit_nodes = extract_graph_summary(graph_def)

	nodes_in_loop = bfs_for_reachable_nodes(exit_nodes, name_to_input_name, name_to_node)
	# import code
	# code.interact(local=locals())

	nodes_in_loop_list = sorted(list(nodes_in_loop), key=lambda n: name_to_seq_num[n])
	# Now construct the output GraphDef
	out = graph_pb2.GraphDef()
	for n in nodes_in_loop_list:
    	# Clean up 'Enter' nodes
		if name_to_node[n].op == 'Enter':
			# import code
			# code.interact(local=locals())
			for item in name_to_node[n].input:
				name_to_node[n].input.remove(item)
				name_to_node[n].op = 'Placeholder'
				dtype = name_to_node[n].attr['T'].type
				name_to_node[n].attr.clear()
				name_to_node[n].attr['dtype'].type = dtype
		out.node.extend([copy.deepcopy(name_to_node[n])])
	out.library.CopyFrom(graph_def.library)
	out.versions.CopyFrom(graph_def.versions)

	return out



# Read out nodes
parser = argparse.ArgumentParser(description='This script generates a quick view of your_frozen_model.pb')
parser.add_argument('your_frozen_model.pb', nargs='+', help='Required argument: your_frozen_model.pb')
args = parser.parse_args()

with tf.gfile.GFile(sys.argv[1], "rb") as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())

loop_subgraph = extract_loops(graph_def)

with tf.Graph().as_default() as graph:
	tf.import_graph_def(loop_subgraph, name='')

for op in graph.get_operations():
	print('op type:', op.type)
	print('op name:', op.name)
	for i in op.inputs:
		print('  in:', i.name, i.shape)
	for i in op.control_inputs:
		print('  control in:', i.name)
	for o in op.outputs:
		print('  out:', o.name, o.shape)
	print('\n')

print('\nNum ops:', len(graph.get_operations()))
