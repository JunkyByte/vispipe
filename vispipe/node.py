from typing import List, Tuple, Callable
from inspect import signature, _empty, getdoc
from functools import partial
from types import FunctionType
from ast import literal_eval
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
    intercept_end: bool
        Whether the block should intercept pipeline end and manually manage termination.
        This is a complex and advanced feature and must be implemented correctly or pipeline
        termination is not assured.
    allow_macro: bool
        Whether the block can be part of macro blocks.
        While this should usually be `True` blocks that intercept the end of pipeline or use
        :meth:`.vispipe.Pipeline._empty` and :meth:`.vispipe.Pipeline._skip` should set this flag `False`
    """
    @staticmethod
    def serialize_args(args: dict):
        return dict((k, Block._stringify(v)) for k, v in args.items())

    @staticmethod
    def _stringify(x):
        if isinstance(x, np.ndarray):
            return np.array2string(x, separator=',')
        return str(x)

    def __init__(self, f: Callable, is_class: bool, max_queue: int, output_names: List[str],
            tag: str, data_type: str, intercept_end: bool, allow_macro: bool):
        self.f = f
        self.name = f.__name__
        self.is_class = is_class
        self.tag = tag
        self.data_type = data_type
        self.intercept_end = intercept_end
        self.allow_macro = allow_macro
        self.docstring = getdoc(self.f)
        if self.is_class:
            init_params = signature(self.f).parameters
            if any([v.default == _empty for v in init_params.values()]):
                raise Exception('Some custom arguments of node <%s> have no default value set' % self.name)
            input_args = {**init_params, **signature(self.f.run).parameters}
            del input_args['self']
        else:
            input_args = dict(signature(self.f).parameters)
        self.input_args = dict([(k, val.default) for k, val in input_args.items()])
        self.input_args_type = dict([(k, type(val)) for k, val in self.input_args.items()])
        self.max_queue = max_queue
        self.output_names = output_names if output_names is not None else ['y']

    def get_function(self, custom_args={}):
        if self.is_class:
            return self.f(**custom_args).run
        return partial(self.f, **custom_args)

    def num_inputs(self):
        return len(self.input_args)

    def num_outputs(self):
        return len(self.output_names)

    def serialize(self):
        x = dict(self)
        del x['f']
        # TODO: Change me to a custom value, using None can cause problems
        x['input_args'] = dict((k, v if v != _empty else None)
                                for k, v in x['input_args'].items())
        x['input_args_type'] = dict((k, str(v.__name__)) for k, v in x['input_args_type'].items())
        x['custom_args'] = Block.serialize_args(x['custom_args']) # TODO
        return x

    def __iter__(self):
        yield 'f', self.f
        yield 'name', self.name
        yield 'input_args', self.input_args
        yield 'custom_inputs_type', self.custom_inputs_type
        yield 'max_queue', self.max_queue
        yield 'output_names', self.output_names
        yield 'tag', self.tag
        yield 'data_type', self.data_type
        yield 'intercept_end', self.intercept_end
        yield 'allow_macro', self.allow_macro
        yield 'docstring', self.docstring


class Node:
    """
    The class used to represent a node of the pipeline.

    Parameters
    ----------
    node_block: Tuple[Block, str]
        The block of the pipeline the node represent and the name used to initialize the node.

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
    name: Optional[str]
        The name of the node, used for output management
    """

    def __init__(self, node_block: Tuple[Block, str], **kwargs):
        self.block = node_block[0]
        self.name = node_block[1]
        self.is_macro = False
        self._custom_args = kwargs
        self.out_queues = []
        self.clear_out_queues()
        self._hash = None

    def clear_out_queues(self):
        self.out_queues = []
        for _ in range(self.block.num_outputs()):
            self.out_queues.append([])

    @property
    def custom_args(self):
        return self._custom_args

    def set_custom_arg(self, key, value):
        arg_type = self.block.custom_args_type[key]
        if arg_type in [list, bool, tuple, dict, None, bytes, np.ndarray]:
            try:
                parsed = literal_eval(value)
                if arg_type is np.ndarray:
                    parsed = np.array(parsed)
                if isinstance(parsed, arg_type):
                    self._custom_args[key] = parsed
                else:
                    raise TypeError('Custom arg "%s" of "%s" with value "%s" is not of type "%s"' %
                            (key, self.block.name, parsed, arg_type))
            except (ValueError, SyntaxError):
                raise ValueError('Cannot parse custom arg "%s" of "%s" with value "%s"' %
                        (key, self.block.name, value)) from None
        elif arg_type is FunctionType:
            print('is a function')
            try:
                self._custom_args[key] = eval(value)
            except (ValueError, SyntaxError):
                raise ValueError('Cannot parse custom arg "%s" of "%s" with value "%s"' %
                        (key, self.block.name, value)) from None
        else:
            self._custom_args[key] = arg_type(value)

    def __hash__(self):
        if self._hash is None:
            return id(self)
        return self._hash

    def __getstate__(self):
        state = self.__dict__.copy()
        state['out_queues'] = [[] for _ in range(self.block.num_outputs())]
        return state
