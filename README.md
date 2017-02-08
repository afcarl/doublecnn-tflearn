# doublecnn-tflearn
TFlearn layer implemnetation for doublecnn


Execute `python tf_mnist_double.py`.


## Problems

```
W_shape: (32, 28, 3, 3)
prod: 252
identity: (252, 252)
new shape: (252, 28, 3, 3)
Traceback (most recent call last):
  File "tf_mnist_double.py", line 23, in <module>
    network = conv_2d_double(network, 32, 3, activation='relu')
  File "/home/moose/GitHub/doublecnn-tflearn/double_cnn.py", line 69, in conv_2d_double
    W_effective = tf.nn.conv2d(W, filter_, strides, padding='VALID')
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_nn_ops.py", line 396, in conv2d
    data_format=data_format, name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.py", line 759, in apply_op
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 2242, in create_op
    set_shapes_for_outputs(ret)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1617, in set_shapes_for_outputs
    shapes = shape_func(op)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1568, in call_with_requiring
    return call_cpp_shape_fn(op, require_shape_fn=True)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 610, in call_cpp_shape_fn
    debug_python_shape_fn, require_shape_fn)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 675, in _call_cpp_shape_fn_impl
    raise ValueError(err.message)
ValueError: Dimensions must be equal, but are 32 and 3 for 'Conv2D/Conv2D' (op: 'Conv2D') with input shapes: [3,3,1,32], [252,28,3,3].
```

