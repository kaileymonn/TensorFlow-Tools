# TensorFlow Graph Surgery Tool #
Built for object detection models, other workloads may not yield desired results. Building on TensorFlow's Graph Transform Tool.

## Introduction ##
Various preprocessing and optimization steps for TensorFlow models specific to Ambarella's toolchain.  

### Usage Examples ###
Quick inspect: Identifies feed/fetch names, feed shapes   
  - $ python3 tf_graph_surgery.py -p path/to/model.pb  
General: Performs -cs, -gt (default arguments), -df in this order
  - $ python3 tf_graph_surgery.py -p path/to/model.pb \  
    -o output_name.pb \  
    -all  
Custom transform arguments: You can specify your own transforms using the -gt flag. -cs, -gt, -df can be used independently and individually. Execution order follows user specified c-line input. 
  - $ python3 tf_graph_surgery.py -p path/to/model.pb \    
    -o output_name.pb \    
    -cs \    
    -gt 'quantize_weights quantize_nodes' \    
    -df
    * Only passing the -gt flag without any values will perform -gt with the default transforms. You can use this if for e.g. -df is problematic and you don't want to use it. Instead of running -all, you can run -cs -gt to mimic -all without the -df processing at the end. 
I/O control: When generating subgraphs, users can specify input and output node names using the -in and -on flags respectively. If -in or -on are unused, their values are defaulted to the feed and fetch nodes of the graph which would be determined internally. 
  - $ python3 tf_graph_surgery.py -p path/to/model.pb \  
    -o output_name.pb \    
    -in "node_a" "node_b" \  
    -all  
    * You will get the optimized subgraph starting from new placeholders "node_a" and "node_b" up to the fetch nodes of the original model. 
  - $ python3 tf_graph_surgery.py -p path/to/model.pb \  
    -o output_name.pb \    
    -in "node_a" \    
    -on "node_c" \    
    -all 
    * You will get the optimized subgraph bounded by "node_a" and "node_c"
  - $ python3 tf_graph_surgery.py -p path/to/model.pb \  
    -o subgraph.pb \    
    -in "node_a" "node_b" \    
    -is "1,5118,4" "5118,1" \    
    -on "node_c" \    
    -all 
    * You will get the optimized subgraph bounded by "node_a","node_b" and "node_c"
    * node_a will be passed the placeholder shape [1,5118,4], node_b will be passed the shape [5118,1]
    * -is is done automatically for you by default but in instances where their shapes cannot be determined, you will be prompted to use -is.
  - $ python3 tf_graph_surgery.py -p subgraph.pb \    
    -o output_name.pb \    
    -fs "1,5118,4" "5118,1" \        
    -all 
    * In this example, our input graph is the output of the previous example, so a subgraph bounded by "node_a", "node_b" and "node_c"
    * We have to now specify feed shapes using the -fs flag since -fs is defaulted to ["1,224,224,3"] which is meant for models that have 1 feed (for the image tensor). 
    * The tool will automatically check the rank of your input shapes when using either -fs or -is and throw error messages if their ranks do not match the required rank for the feed nodes or specified input nodes. 
Very custom use: For users very comfortable with TensorFlow's Graph Transform Tool and faced with a unique model, they may choose to run specialized sequences of transforms.
  - $ python3 tf_graph_surgery.py -p path/to/model.pb \  
    -o output_name.pb \    
    -in "node_a" "node_b" \    
    -is "1,5118,4" "5118,1" \    
    -on "node_c" \    
    -gt 'strip_unused_nodes(name=node_a, type_for_name=float, shape_for_name="1,5118,4", name=node_b, type_for_name=float, shape_for_name"5118,1") \    
    -cs \    
    -gt 'remove_nodes(op=Identity op=CheckNumerics op=Assert) fold_constants fold_batch_norms remove_attributes(attribute_name=_class)' \    
    -df   

### Arguments ###
* -p : This is a required argument reflecting the path to your TensorFlow model file.    
* -o : This is an optional argument to set the output TensorFlow protobuf's name.    
* -fs : This is an optional argument to set the dimensions for feed nodes (feed nodes are not necessarily the same as input nodes).    
* -in : This is an optional argument to specify input node names.  
* -is : This is an optional argument to specify dimensions for input nodes.  
* -on : This is an optional argument to specify output node names.  
* -cs : Optional flag to run utility_library.process_shapes().
* -gt : Optional flag to run TensorFlow Graph Transform with arguments defaulted to BASE_TRANSFORMS in tf_graph_surgery.py. See TensorFlow documentation for guides to using their built-in transforms.
* -df : Optional flag to convert the model's default NHWC ordering to NCHW.  
* -all : Optional flag for convenient execution of -cs, -gt, -df in that order, with BASE_TRANSFORMS for -gt.

### Files ###
- tf_graph_surgery.py
- graph_transform.py
- utility_library.py