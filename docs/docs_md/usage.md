# Usage

A bunch of examples of the different features of the package.
Refer to the documentation to get a better view of each function arguments.

### Blocks
A block is a function tagged with the decorator called (guess what) `block`.
```python
from vispipe import block

@block
def identity_block(x):
    return x
```

Or a class with a `run` method.
```python
@block
class identity_class_block:
    def __init__(self):
	# you can have attributes stored and used by run method
    	self.useless_value = 42
	pass

    def run(self, x):
        return x
```

A block can have multiple inputs.

```python
@block
def add(x, y):
    return x + y

# Or none
@block
def just_a_one():
    return 1
```

All the inputs we defined right now are 'real' inputs and will be filled
by connecting the block to other blocks of the pipeline.
We may want to have static arguments as well, an input will become a static argument once we assign a default value to it. If you want to use the visualization you should also specify the type so that they can be parsed correctly.

```python
@block
def add_constant(x, k: int = 42):
    # x will be a 'real' input while k will be a static argument
    return x + k
```

Now that you know how blocks work let's see how to create a pipeline, add and connect them.

### Pipeline Building

```python
from vispipe import Pipeline

# Create a pipeline
p = Pipeline()

# Add nodes by name
ones = p.add_node('just_a_one')
adder = p.add_node('add_constant', k = 10)  # We can specify the value of static arguments during add_node
# The add_node function will return the unique identifier for the node, it is an integer and
# corresponds to the hash of the node, you will use it to interact with the node.

# Connect nodes
# syntax: add_conn(from_hash, output_index, to_hash, input_index)
p.add_conn(ones, 0, adder, 0)

# We have a working pipeline now! Let's start it
p.run()
# We can wait for it to end using
p.join(timeout=1)  # It supports timeout argument similarly to Threading std library

# In this case the pipeline will run indefinitely and we have not way to interact with it.
# Let's Add an output to it
p.add_output(adder)

# If we now run it we can iterate over the outputs of adder
p.run()
for x in p.outputs[adder]:
    print(x)

>>> 11  # It will add 10 that is our constant to 1 which is the just_a_one output.
# (...)
>>> 11  # It will run indefinitely as there's no ending.
```

### Saving and reloading

```python
# Once we are happy with our pipeline we can save it to pickle
p.save(file_path)

# And reload it in a later stage
p = Pipeline()
p.load(file_path)

# Or more concisely
p = Pipeline(path=file_path)
```

### Advanced Block creation
#### `Pipeline._empty` and `Pipeline.skip` objecs
You may have noticed that the flexibility of the blocks we created is pretty limited, we need to return a value at each call and we will always receive an input.
To overcome this there are two particular objects that get treated in a particular way:

`Pipeline._empty` allows to specify that we do not want to return any result yet.
```python
# The example is based on the benchmark block from ops/utils.py
@block
class benchmark:
    def __init__(self):
	self.n = 1000
	self.start_time = None

    def run(self, x):
	if self.start_time is None:
	    self.start_time = time.time()
	
	self.n -= 1
	if self.n == -1:  # After 1000 iterations we return delta time
	    delta = time.time() - self.start_time
	    return delta
	
	# (...) missing code to manage the ending
	
	return Pipeline._empty  # Otherwise we are not ready to return an output
```

`Pipeline._skip(value)` allows to return a value while also skipping the next input.
This is particularly useful when you need to iterate over an input.
```python
# The example is based on iterator from ops/flows.py
@block
class iterator:
    def __init__(self):
	self.iterator = None

    def run(self, x):
	if self.iterator is None:
	    self.iterator = iter(x)

	try:
	    # As we can still iterate we return a skip object containing the next value
	    y = Pipeline._skip(next(self.iterator))
	except StopIteration:
	    # If we finished the iterator we return an empty so that we can wait for next input
	    self.iterator = None
	    y = Pipeline._empty

	return y
```

### Macro Blocks
Macro blocks are a convenient way to speed up a set of linearly connected blocks.
Blocks that are part of a macro will be run together (instead of connected with queues).
While this limits the flexibility of a part of the pipeline the functions will run a lot faster as they completely
skip the communication overhead.
(Please refer to documentation for a better explanation of this functionality)

```python
# (...)
p.add_macro(start_hash, end_hash)  # Will add a macro from start hash to end hash.

p.remove_macro(node_hash) # Will delete the macro the node belongs to.
```

### An example
Loads the mnist dataset from a numpy array + the labels associated.
It then reshape the images to be actual `(28, 28)`, resize them to another resolution
and creates batches from them.
```python
# Create pipeline
p = Pipeline()

# Create nodes
load_images = p.add_node('np_iter_file', path='tests/data/mnist.npy')
load_labels = p.add_node('np_iter_file', path='tests/data/mnist_labels.npy')
reshape = p.add_node('np_reshape', shape=(28, 28))
resize = p.add_node('resize_cv2', width=56, height=56)
batch_images = p.add_node('batchify/images', size=32)
batch_labels = p.add_node('batchify/labels', size=32)

# Add connections
p.add_conn(load_images, 0, reshape, 0)
p.add_conn(reshape, 0, resize, 0)
p.add_conn(resize, 0, batch_images, 0)
p.add_conn(load_labels, 0, batch_labels, 0)

# Set outputs
p.add_output(batch_images)
p.add_output(batch_labels)

# Run it
p.run(slow=False, use_mp=False)
for batch_x, batch_y in zip(p.outputs['images'], p.outputs['labels']):
    print(batch_x.shape, batch_y.shape)
```
