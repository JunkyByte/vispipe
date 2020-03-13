from typing import List, Callable
from inspect import signature, _empty
import numpy as np


class Block:
    """
    The class used to represent one generator function (or class) that has been tagged with the
    block decorator provided by :func:`.vispipe.block`.

    Parameters
    ----------
    f : Callable
        The function (or class with run function) this block represent.
    is_class : bool
        If the function tagged is actually a class.
    max_queue : int
        The maximum size of the queues of this block.
    output_names : List[str]
        The names used by the outputs of your function. The length of this list is used as the number
        of outputs of the function.
    tag : str
        A string tag used to group the blocks (useful during visualization).
    data_type : str
        If a visualization block this field is used to specify the kind of data you want to visualize.
        Check :class:`.vispipe.Pipeline` for a list of the supported types.
    """
    @staticmethod
    def serialize_args(args: dict):
        return dict((k, Block._stringify(v)) for k, v in args.items())

    @staticmethod
    def _stringify(x):
        if isinstance(x, np.ndarray):
            return np.array2string(x, separator=',')
        return str(x)

    def __init__(self, f: Callable, is_class: bool, max_queue: int, output_names: List[str], tag: str, data_type: str):
        self.f = f
        self.name = f.__name__
        self.is_class = is_class
        self.tag = tag
        self.data_type = data_type
        if self.is_class:
            init_params = signature(self.f).parameters
            if any([v.default == _empty for v in init_params.values()]):
                raise Exception('Some custom arguments of node <%s> have no default value set' % self.name)
            input_args = {**init_params, **signature(self.f.run).parameters}
            del input_args['self']
        else:
            input_args = dict(signature(self.f).parameters)
        args = [(k, val.default) for k, val in input_args.items()]
        self.input_args = dict([(k, val) for k, val in args if val is _empty])
        self.custom_args = dict([(k, val) for k, val in args if val is not _empty])
        self.custom_args_type = dict([(k, type(val)) for k, val in self.custom_args.items()])
        self.max_queue = max_queue
        self.output_names = output_names if output_names is not None else ['y']

    def num_inputs(self):
        return len(self.input_args)

    def num_outputs(self):
        return len(self.output_names)

    def serialize(self):
        x = dict(self)
        del x['f']
        # TODO: Change me to a custom value, using None can cause problems
        # TODO: Support np array by casting to nested lists
        x['input_args'] = dict((k, v if v != _empty else None)
                                for k, v in x['input_args'].items())
        x['custom_args_type'] = dict((k, str(v.__name__)) for k, v in x['custom_args_type'].items())
        x['custom_args'] = Block.serialize_args(x['custom_args'])
        return x

    def __iter__(self):
        yield 'f', self.f
        yield 'name', self.name
        yield 'input_args', self.input_args
        yield 'custom_args', self.custom_args
        yield 'custom_args_type', self.custom_args_type
        yield 'max_queue', self.max_queue
        yield 'output_names', self.output_names
        yield 'tag', self.tag
        yield 'data_type', self.data_type


class Node:
    """
    The class used to represent a node of the pipeline.

    Parameters
    ----------
    node_block: Block
        The block of the pipeline the node represent.

    Attributes
    ----------
    block: Block
        The block of the the pipeline the node represent.
    custom_args: dict
        The custom arguments used for calling the block funtion during execution.
    _hash: int
        Custom hash used to reload the pipeline checkpoints.
    out_queues: List[Queue]
        The list of output queues used to process the outputs of the node.
    """

    def __init__(self, node_block: Block, **kwargs):
        self.block = node_block
        self.custom_args = kwargs
        self._hash = None
        self.out_queues = []
        self.clear_out_queues()

    def clear_out_queues(self):
        self.out_queues = []
        for _ in range(self.block.num_outputs()):
            self.out_queues.append([])

    def __hash__(self):
        if self._hash is None:
            return id(self)
        return self._hash

    def __getstate__(self):
        state = self.__dict__.copy()
        state['out_queues'] = [[] for _ in range(self.block.num_outputs())]
        return state
