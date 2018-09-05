#!/usr/bin/env python3

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exposes the Python wrapper for graph transforms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
# pylint: disable=unused-import,wildcard-import, line-too-long
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.pywrap_tensorflow import TransformGraphWithStringInputs
from tensorflow.python.util import compat

def TransformGraph(input_graph_def, inputs, outputs, transforms):
    """Python wrapper for the Graph Transform Tool.

    Gives access to all graph transforms available through the command line tool.
    See documentation at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
    for full details of the options available.

    Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    inputs: List of node names for the model inputs.
    outputs: List of node names for the model outputs.
    transforms: List of strings containing transform names and parameters.

    Returns:
    New GraphDef with transforms applied.
    """

    input_graph_def_string = input_graph_def.SerializeToString()
    inputs_string = compat.as_bytes(inputs)
    outputs_string = compat.as_bytes(outputs)
    transforms_string = compat.as_bytes(transforms)
    print(inputs_string)
    print(outputs_string)
    print(transforms_string)
    with errors.raise_exception_on_not_ok_status() as status:
        output_graph_def_string = TransformGraphWithStringInputs(
                                  input_graph_def_string, inputs_string, outputs_string,
                                  transforms_string, status)
    output_graph_def = graph_pb2.GraphDef()
    output_graph_def.ParseFromString(output_graph_def_string)
    return output_graph_def

def main(FLAGS):

    ##################### start optimization ###################################
    graph_def = tf.GraphDef()
    with open(FLAGS.input_graph, "rb") as f:
        graph_def.ParseFromString(f.read())

    output_def = TransformGraph(graph_def, FLAGS.input_node_names, FLAGS.output_node_names, FLAGS.transforms)

    ############## save the optimized graph #######################
    tf.train.write_graph(output_def, '.', FLAGS.output_graph, as_text=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input_graph",
      type=str,
      default="",
      help="TensorFlow \'GraphDef\' file to load.")
  parser.add_argument(
      "--output_graph",
      type=str,
      default="",
      help="Output \'GraphDef\' file name.")
  parser.add_argument(
      "--input_node_names",
      type=str,
      default="",
      help="The name of the input nodes, comma separated.")
  parser.add_argument(
      "--output_node_names",
      type=str,
      default="",
      help="The name of the output nodes, comma separated.")
  parser.add_argument(
      "--transforms",
      type=str,
      default="",
      help="Transforms to be applied, comma separated.")

  FLAGS = parser.parse_args()
  main(FLAGS)
