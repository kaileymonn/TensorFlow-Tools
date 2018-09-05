# TensorFlow Preprocessing Tools #
Built for object detection models, other workloads may not be ideal. Building on TensorFlow's Graph Transform Tool. This is a WIP

## Introduction ##
Various preprocessing utilities and optimization tools for TensorFlow models. Found in /Preprocessor.

### Quick start ###
Common usage involves running the following commands:   
* $ python3 tf_preprocessor.py -p path/to/model.pb 

### Arguments ###
* -p : This is a required argument reflecting the path to your TensorFlow model file  
* -o : This is an optional argument to set the output TensorFlow protobuf's name
* -is : This is an optional argument to set the input image's dimensions
* -in : This is an optional argument to specify input nodes
* -on : This is an optional argument to specify output nodes
* -T : This is an optional argument to specify transforms. If flag is not used, default transforms will be 'fold_constants fold_batch_norms fold_old_batch_norms merge_duplicate_nodes remove_nodes(op=Identity op=CheckNumerics op=Assert) strip_unused_nodes remove_attribute(attribute_name=_class) sort_by_execution_order'
* -f : This is an optional argument to convert TensorFlow model's default NHWC ordering to NCHW 

### Files ###
- tf_preprocessor.py
- utility_library.py
