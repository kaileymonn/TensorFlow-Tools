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

import sys
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='This script generates a quick view of your_frozen_model.pb')
parser.add_argument('your_frozen_model.pb', nargs='+', help='Required argument: your_frozen_model.pb')
args = parser.parse_args()

with tf.gfile.GFile(sys.argv[1], "rb") as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
	tf.import_graph_def(graph_def, name='')

for op in graph.get_operations():
	if op.name == 'ToFloat_3':
		import code
		code.interact(local=locals())
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
