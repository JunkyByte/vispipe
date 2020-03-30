# VisPipe
## Build Pipelines visually and run them via code
[![CodeFactor](https://www.codefactor.io/repository/github/junkybyte/vispipe/badge/master?s=b4f0ed72fedffa8ed8cbc9bc9887a0db528a24b2)](https://www.codefactor.io/repository/github/junkybyte/vispipe/overview/master)
![Package Testing](https://github.com/JunkyByte/vispipe/workflows/Package%20Testing/badge.svg?branch=master)

VisPipe is a python package that can help you build pipelines by providing a convenient visual creation tool that can simplify the construction and debugging of otherwise blackbox pipelines.
Once you are satisfied with the result a pipeline can be saved to file and run via code, the outputs of it can be easily interacted with through python code.

By default VisPipe provides a number of Operational blocks, you are encouraged to extend them by creating your own.
VisPipe will run using python `Threads` or `Process` (+ `Queues`) internally to reduce the tradeoff between performance and flexibility.
Each block of your pipeline will support multiple input/output arguments, multiple connections and custom static arguments.

## Installation
```
git clone https://github.com/JunkyByte/vispipe.git
pip install vispipe
```

## Usage
A bunch of examples of the different features of the package.
Refer to the documentation #TODO ADD LINK# to get a better view of each function arguments.

### Blocks
A block is a generator function with a single shot yield tagged with the decorator called (guess what) `block`.
```python
from vispipe import block

@block
def identity_block(x):
    yield x
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
        yield x
```

A block can have multiple inputs.

```python
@block
def add(x, y):
    yield x + y

# Or none
@block
def just_a_one():
    yield 1
```

All the inputs we defined right now are 'real' inputs and will be filled
by connecting the block to other blocks of the pipeline.
We may want to have static arguments as well, an input will become a static argument once we assign a default value to it. If you want to use the visualization you should also specify the type so that they can be parsed correctly.

```python
@block
def add_constant(x, k: int = 42):
    # x will be a 'real' input while k will be a static argument
    yield x + k
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
