from .node import Node, Block
from .graph import Graph
from functools import partial, reduce
from itertools import chain
from inspect import signature, isgeneratorfunction
from typing import Callable, Optional, List, Union, Tuple, Any
import queue
import multiprocessing.queues as mpqueues
import multiprocessing as mp
import threading
from ast import literal_eval
import numpy as np
import types
import copy
import pickle
import time
import logging
MAXSIZE = 100
log = logging.getLogger('vispipe')
log.setLevel(logging.DEBUG)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
log.addHandler(console_handler)

# TODO: Check save load works with outputs
# TODO: Add drag to resize (or arg to resize) to vis blocks as you can now zoom.
# TODO: Blocks with inputs undefined? Like tuple together all the inputs, how to?
# TODO: Variable output size based on arguments.
# TODO: during vis redirect console to screen?
# TODO: Add some assertion on ckpt reloading -> If a block is part of macro but not is not allow_macro for ex.
# TODO: Add some custom exceptions
# TODO: Add macro block setting during visualization.


class Pipeline:
    """
    Pipeline class that is used to represent pipelines.

    Parameters
    ----------
        path : Optional[str]
            The path from which load a pipeline checkpoint.
            If not specified an empty pipeline is created.

    Attributes
    ----------
    pipeline: Graph
    runner: PipelineRunner
    vis_mode: bool
        Whether the pipeline is in visualization mode.
        This is filled during checkpoint loading.
    macro: List[List[int]]
        The list of all the macro blocks.
        Each macro is represented as a list of node hashes.
    """

    #: Yield this to specify that a block is not ready to create a new output.
    _empty = object()

    #: Yield an instance of this to specify that a block is busy processing old input, on next call no new data will be passed.
    class _skip:
        def __init__(self, x):
            self.x = x

    data_type = ['raw', 'image', 'plot']  #: Supported data_type for visualization blocks.
    vis_tag = 'vis'  #: The tag used for visualization blocks.

    #: The maximum size of the output queues. This is set to avoid out of memory with running pipelines.
    MAX_OUT_SIZE = 1000

    #: Whether to use a warning wrapper for blocks that are part of macro blocks.
    #: Setting this to `False` will speed up pipelines that use macro blocks but will make debugging more difficult.
    USE_MACRO_WARNINGS = True

    _blocks = {}

    def __init__(self, path: Optional[str] = None):
        self.pipeline = Graph(MAXSIZE)
        self.runner = PipelineRunner()
        self._outputs = []
        self.macro = []
        self.vis_mode = False

        if path:
            self.load(path)

    @property
    def nodes(self) -> List[Node]:
        """
        Returns the list of :class:`.Node` that are part of the pipeline.

        Returns
        -------
        List[Node]:
            The list of nodes.
        """
        return self.pipeline.v()

    @property
    def outputs(self) -> dict:
        """
        Returns a dictionary that maps the output node name to the output node iterator class.
        If the name of the node was not specified (it can happen if you use hashes to create
        outputs) the mapping will use the hash as name.

        Returns
        -------
        dict:
            Dictionary containing the outputs of the pipeline.
        """
        if not self.runner.built:
            raise Exception('The pipeline has to be run before accessing its outputs')

        out = {}
        for k, v in self.runner.outputs.items():
            name = self.get_node(k).name
            out[name if name else k] = v
        return out

    @staticmethod
    def register_block(func: Callable, is_class: bool, max_queue: int, output_names: List[str],
            tag: str, data_type: str, intercept_end: bool, allow_macro: bool) -> None:
        """
        Register a block in the pipeline list of available blocks, this function is called
        automatically by the decorator.

        Note
        ----
        This is a static method, if you want to refer to it you can use ``Pipeline.register_block``

        Parameters
        ----------
        func : Callable
            The function to be registered.
        is_class : bool
            Whether it is a class function.
        max_queue : int
            The max size of the queues used for the block.
        output_names : List[str]
            See :class:`.Block`
        tag : str
            See :class:`.Block`
        data_type : str
            See :class:`.Block`
        allow_macro: bool
            See :class:`.Block`
        """
        block = Block(func, is_class, max_queue, output_names, tag, data_type, intercept_end, allow_macro)
        assert block.name not in Pipeline._blocks.keys(), 'The name %s is already registered as a pipeline block' % block.name
        Pipeline._blocks[block.name] = block

    @staticmethod
    def get_blocks(serializable: bool = False) -> List[dict]:
        """
        Returns all the blocks tagged as a dictionary.

        Note
        ----
        This is a static method, if you want to refer to it you can use ``Pipeline.get_blocks``

        Parameters
        ----------
        serializable : bool
            Whether if the dictionary returned must be serializable.

        Returns
        -------
        List[dict]:
            A list of dictionaries representing each block.
        """
        blocks = []
        for block in Pipeline._blocks.values():
            if serializable:
                block = block.serialize()
            else:
                block = dict(block)
            blocks.append(block)
        return blocks

    def get_output(self, node: Union[int, str]):
        """
        Returns the output node iterator class with name or hash you specified.

        Note
        -----
        While its suggested to use `Pipeline.outputs[node]` to access a particular output
        this method can be convenient if you want to access a node by its hash even if it
        hash a specified name (in the outputs you can access it only by name).

        Parameters
        ----------
        node : Union[int, str]
            The hash or name of the node you are looking for.
        """
        if isinstance(node, str):
            node = hash(self.get_node(node))
        return self.runner.outputs[node]

    def connections(self, node_hash: int, out=None) -> List[Tuple]:
        """
        Returns the list of connections of a particular node.
        Each connections is a tuple of type ``(other_hash, out_idx, inp_idx, bool)``
        where bool is ``True`` if the connection is to ``other``, ``False`` otherwise.

        Parameters
        ----------
        node_hash : int
            The hash of the node.
        out : Union[bool, None]
            If ``None`` returns all the connections of the node.
            If ``True`` returns all the output connections.
            If ``False`` returns all the input connections.

        Returns
        -------
        List[Tuple]:
            The list of connections.
        """
        node = self.get_node(node_hash)
        return self.pipeline.adj(node, out)

    def get_node(self, node: Union[int, str]):  # While the hash is unique the name may not be
        """
        Returns the node that corresponds to name or hash.

        Note
        ----
        Names are arbitrary and unicity is NOT verified.
        Using name to access nodes may lead to undesired results if the name is duplicated.

        Parameters
        ----------
        node : Union[int, str]
            The name or hash of the node you are looking for.
        """
        return self.pipeline.get_node(node)

    def add_node(self, block_name: str, **kwargs) -> int:
        """
        Adds a new node to the pipeline.

        Parameters
        ----------
        block_name : str
            The name of the block you want to add.
            You can append `'/name'` to specify a name that will be used to identify the node as an output.
            For example `'sin/sin_preprocessing'` is a valid name for a node called `sin` with name
            initialized to `'sin_preprocessing'`
        **kwargs
            Used to specify custom arguments for the block.

        Note
        ----
        Names are arbitrary and unicity is NOT verified.
        Creating multiple nodes with same name can produce unexpected results.
        Be sure to use unique names for output nodes.

        Returns
        -------
        int:
            The hash of the created node
        """
        if '/' in block_name:
            block_name, name = block_name.split('/')
        else:
            name = None

        node = Node((Pipeline._blocks[block_name], name), **kwargs)
        return self.pipeline.insertNode(node)

    def remove_node(self, node_hash: Union[str, int]):
        """
        Remove a node from the pipeline.

        Parameters
        ----------
        node : Union[str, int]
            The name or hash of the node you want to remove.
        """
        node = self.get_node(node_hash)
        if isinstance(node, str):
            node_hash = hash(node)

        if node in self._outputs:
            self._outputs.remove(node)

        try:
            self.remove_macro(node_hash)
        except KeyError:  # If the node is not part of a macro block will throw a KeyError
            pass

        self.pipeline.deleteNode(node)

    def add_output(self, output: Union[str, int]):
        """
        Mark a node as output, outputs will be the entry point of the pipeline during execution.

        Note
        ----
        If an output is not consumed during pipeline execution it can stall once its queue
        is full. Ideally all nodes marked as output are consumed.

        Parameters
        ----------
        output : Union[str, int]
            The name or hash of the node you want to mark as an output.
        """
        node = self.get_node(output)
        if isinstance(output, str):
            output = hash(node)

        if output in self._outputs:
            raise Exception('The node specified is already an output')
        if node.block.tag == Pipeline.vis_tag:
            raise Exception('Visualization blocks cannot be used as outputs')

        is_macro, index = self.is_macro(output)
        if is_macro and not index:
            raise Exception('Only last node of a macro block can be an output')

        self._outputs.append(output)

    def remove_output(self, output: Union[str, int]):
        """
        Remove an output from the pipeline.

        Parameters
        ----------
        output : Union[str, int]
            The name or hash of the node you want to remove from the list of outputs.
        """
        if isinstance(output, str):
            node = self.get_node(output)
            output = hash(node)
        self._outputs.remove(output)

    def remove_tag(self, tag: str):
        """
        Remove all nodes with a particular tag.

        Parameters
        ----------
        tag : str
            The tag you want to remove.
        """
        hashes = [hash(node) for node in self.nodes if node.block.tag == tag]
        for node_hash in hashes:
            self.remove_node(node_hash)

    def add_conn(self, from_hash: int, out_index: int, to_hash: int, inp_index: int):
        """
        Add a connection between two nodes of the pipeline.
        An output can have multiple connections while an input can have only one.

        Parameters
        ----------
        from_hash : int
            The hash of the output node you are connecting.
        out_index : int
            The index of the output you want to connect (starting from zero).
        to_hash : int
            The hash of the input node you are connecting to.
        inp_index : int
            The index of the input you want to connect (starting from zero).
        """
        from_is_macro, from_m_index = self.is_macro(from_hash)
        to_is_macro, to_m_index = self.is_macro(from_hash)
        if from_is_macro and not from_m_index:
            raise Exception('The output node is part of a macro block, only last node of a macro can have new connections')
        if to_is_macro and to_m_index is not False:
            raise Exception('The input node is part of a macro block, only first node of a macro can have new connections')
        from_node = self.get_node(from_hash)
        to_node = self.get_node(to_hash)
        self.pipeline.insertEdge(from_node, to_node, out_index, inp_index)

    def add_macro(self, start_hash: int, end_hash: int) -> None:
        """
        Create a macro block from `start_hash` to `end_hash`, macro blocks will be executed in a faster way internally
        Only a linearly connected set of nodes can become a macro block.
        Each block must have `allow_macro` set to `True` to support being part of a macro block.
        Only the last node of a macro block can be marked as an output and every node intern to a macro cannot have
        outside connections.
        `Pipeline._empty` and `Pipeline._skip` of intern nodes will not be correctly handled and an exception will be
        raised if encountered.

        Parameters
        ----------
        start_hash : int
            The node that starts the macro block.
        end_hash : int
            The node that ends the macro block.
        """
        if start_hash == end_hash:
            raise Exception('start and end node coincide')
        start_node = self.get_node(start_hash)
        end_node = self.get_node(end_hash)
        if not end_node.block.allow_macro:  # Manually check end node status
            raise Exception('End node %s has "allow_macro" set to False and cannot be part of a macro block')

        visited = []  # Will contain all hashes that are part of the macro block
        current_node = start_node
        while current_node not in visited:
            # We need to check that the nodes are linearly connected.
            # For simplicity we want that each node is connected ONLY to the subsequent one.
            visited.append(hash(current_node))
            if not current_node.block.allow_macro:
                raise Exception('Block "%s" has "allow_macro" set to False and cannot be part \
                        of a macro block' % current_node.block.name)

            # Checks on connections must be made BEFORE the node switches to new one
            conn_nodes = set(adj[0] for adj in self.pipeline.adj(current_node, out=True))
            if not conn_nodes:
                raise Exception('The two nodes are not connected and cannot be part of the same macro block')

            if len(conn_nodes) > 1 and current_node is not end_node:
                raise Exception('Block "%s" is connected to multiple blocks and cannot be part \
                        of a macro block' % current_node.block.name)

            in_conn_nodes = set(adj[0] for adj in self.pipeline.adj(current_node, out=False))
            if len(in_conn_nodes) > 1 and current_node is not start_node:
                raise Exception('Block "%s" is connected to multiple blocks and cannot be part \
                        of a macro block' % current_node.block.name)

            # We can now switch to next node
            current_node = conn_nodes.pop()
            if current_node.block.intercept_end and current_node is not start_node:
                raise Exception('Only first node of a macro block is allowed to intercept the end of the pipeline')

            if current_node.is_macro:
                raise Exception('The node %s is already part of a macro block.' % hash(current_node))

            if current_node is end_node:  # We reached last node
                break  # Exit the cycling

            if hash(current_node) in self._outputs:
                raise Exception('The node %s is already an output, only last node of a macro block is allowed to be an output.')

        visited.append(hash(end_node))
        for node in [self.get_node(n) for n in visited]:
            node.is_macro = True

        self.macro.append(visited)

    def is_macro(self, node_hash: int) -> Tuple[bool, Optional[bool]]:
        """
        Whether a block is part of a macro block.

        Parameters
        ----------
        node_hash : int
            The node you want to verify.

        Returns
        -------
        Tuple[bool, Optional[bool]]:
            The first boolean represent whether the block is part of a macro block.
            The second one is:
            `None` is the block is internal to its macro
            `False` if is first block of its macro and
            `True` if is last block of its macro.
        """
        for m in self.macro:
            for i, _ in enumerate(m):
                if i + 1 == len(m):
                    index = True
                elif i == 0:
                    index = False
                else:
                    index = None
                return True, index
        return False, None

    def remove_macro(self, node_hash: int) -> None:
        """
        Remove the macro block the node is part of.

        Parameters
        ----------
        node_hash : int
            The node whose macro block you want to remove.
        """
        for m in self.macro:
            if node_hash in m:
                self.macro.remove(m)
                return
        raise KeyError('The node you passed is not part of a macro')

    def set_custom_arg(self, node_hash: int, key: str, value: Any):
        """
        Set a custom argument of a node.

        Parameters
        ----------
        node_hash : int
            The hash of the node.
        key : str
            The name of the argument you want to set.
        value : Any
            The value to set for this argument.
        """
        node = self.get_node(node_hash)
        arg_type = node.block.custom_args_type[key]
        if arg_type in [list, bool, tuple, dict, None, bytes, np.ndarray]:
            try:
                parsed = literal_eval(value)
                if arg_type is np.ndarray:
                    parsed = np.array(parsed)

                if isinstance(parsed, arg_type):
                    node.custom_args[key] = parsed
                else:
                    raise TypeError('Custom arg "%s" of "%s" with value "%s" is not of type "%s"' %
                            (key, node.block.name, parsed, arg_type))
            except (ValueError, SyntaxError):
                raise ValueError('Cannot parse custom arg "%s" of "%s" with value "%s"' %
                        (key, node.block.name, value)) from None
        else:
            node.custom_args[key] = arg_type(value)

    def clear_pipeline(self):
        """
        Resets the state of the pipeline by deleting all its nodes and clearing its outputs.
        """
        self.pipeline.resetGraph()
        self._outputs = []
        self.macros = []
        self.runner.unbuild()

    def read_vis(self):
        """
        Reads the current data passing from all the visualization blocks.

        Returns
        -------
        dict:
            The current visualization data.
        """
        return self.runner.read_vis()

    def read_qsize(self):
        """
        Reads the current queues size from all the blocks.

        Returns
        -------
        List[Tuple]:
            A list of tuples of form (hash, block_name, current_size, max_size).
        """
        if not self.runner.built:
            return []

        def _unpack(h, n, x) -> list:
            if isinstance(x, (queue.Queue, mpqueues.Queue, QueueConsumer)):
                return (h, n, x.qsize(), q.maxsize)
            if x and isinstance(x, list):
                if len(x) == 2 and isinstance(x[1], bool):
                    return (h, n, x[0].qsize(), x[0].maxsize)
                x = [_unpack(h, n, v) for v in x]
                return x

        nodes_queues = []
        for n in self.nodes:
            if isinstance(n.out_queues, list) and not any(n.out_queues):
                continue

            if isinstance(n.out_queues, (queue.Queue, mpqueues.Queue, QueueConsumer)):
                q = n.out_queues
                nodes_queues.append([(hash(n), n.block.name, q.qsize(), q.maxsize)])
            else:
                nodes_queues.extend([_unpack(hash(n), n.block.name, q) for q in n.out_queues])
        return list(chain.from_iterable(nodes_queues))

    def build(self, use_mp: bool) -> None:
        """
        Note
        ----
            This will automatically be called by running the pipeline.
            You should not need to call this method.

        Build the pipeline and make it ready to be run.
        It creates the list of associated :class:`.TerminableThread` and :class:`.QueueConsumer`
        and binds them together using `CloseableQueue`

        To build the pipeline graph we follow the `Kahn's Algorithm for topologic sorting
        <https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm>`_
        applied to the nodes that have their inputs satisfied.

        Pseudo Code (from link)::

            ----- Kahn's algorithm for topologic sorting. -----
            L ← Empty list that will contain the sorted elements
            S ← Set of all nodes with no incoming edge

            while S is non-empty do
                remove a node n from S
                add n to tail of L
                for each node m with an edge e from n to m do
                    remove edge e from the graph
                    if m has no other incoming edges then
                        insert m into S

            if graph has edges then
                return error   (graph has at least one cycle)
            else
                return L   (a topologically sorted order)

        use_mp : bool
            Whether to use multiprocessing instead of threading.
        """
        self.runner.build_pipeline(self.pipeline, self.macro, self._outputs, self.vis_mode, use_mp)

    def run(self, slow: bool = False, use_mp: bool = False) -> None:
        """
        Run the pipeline.

        Parameters
        ----------
        slow : bool
            Whether to run the pipeline in slow mode, useful for visualization and debugging.
        use_mp: bool
            Whether to use multiprocessing instead of threading.
        """
        self.stop()
        self.build(use_mp)
        self.runner.run(slow)

    def stop(self) -> None:
        """
        Stops the pipeline.
        """
        if self.runner.built:
            self.runner.stop()

        for node in self.nodes:
            node.clear_out_queues()
        self.runner.unbuild()

    def save(self, path: str, vis_data: dict = {}) -> None:
        """
        Save the pipeline to file.

        Parameters
        ----------
        path : str
            The path of the file you want to save the pipeline to.
        vis_data : dict
            The visualization data associated with this pipeline.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.pipeline, self._outputs, self.macro, vis_data), f)

    def load(self, path: str, vis_mode: bool = False, exclude_tags: list = []) -> object:
        """
        Loads a pipeline checkpoint from a previously created `.pickle` file.

        Parameters
        ----------
        path : str
            The path to the pickle file.
        vis_mode : bool
            Visualization mode, can cause memory leaks if no visualization is actually connected.
            If `False` all the visualization blocks will be automatically removed. For this reason you should not save
            the pipeline once you loaded it or you will lose the visualization part.
        exclude_tags : list
            Tags to exclude from the checkpoint. All the nodes with these tags will be removed from the pipeline.

        Returns
        -------
        object:
            The visualization data loaded in the checkpoint, you can ignore it if you are not using visualization.
        """
        self.vis_mode = vis_mode
        if not vis_mode:
            exclude_tags.append(Pipeline.vis_tag)

        self.clear_pipeline()
        with open(path, 'rb') as f:
            self.pipeline, self._outputs, self.macro, vis_data = pickle.load(f)

        for tag in exclude_tags:
            self.remove_tag(tag)

        return vis_data

    def join(self, timeout: Optional[float] = None) -> None:
        """
        This is builded following `Thread.join` method from `Threading` module.
        `Refer to the original documentation <https://docs.python.org/3.7/library/multiprocessing.html?#multiprocessing.Process.join>`_
        which is reported in the following note.

        Note
        ----
            Wait until the thread terminates.

            This blocks the calling thread until the thread whose join() method is
            called terminates -- either normally or through an unhandled exception
            or until the optional timeout occurs.

            When the timeout argument is present and not None, it should be a
            floating point number specifying a timeout for the operation in seconds
            (or fractions thereof). As join() always returns None, you must call
            is_alive() after join() to decide whether a timeout happened -- if the
            thread is still alive, the join() call timed out.

            When the timeout argument is not present or None, the operation will
            block until the thread terminates.

            A thread can be join()ed many times.

            join() raises a RuntimeError if an attempt is made to join the current
            thread as that would cause a deadlock. It is also an error to join() a
            thread before it has been started and attempts to do so raises the same
            exception.
        """
        for thr in self.runner.threads:
            thr.join(timeout)

    def is_alive(self) -> bool:
        """
        Returns whether the pipeline is alive.
        The pipeline is considered alive if at least one of its nodes is running.
        """
        return any(thr.is_alive() for thr in self.runner.threads)


class PipelineRunner:
    """
    The class used to build and run pipelines.

    Attributes
    ----------
    built: bool
        Whether the pipeline has already been built and can be run or not.
    threads: List[TerminableThread]
        The list of the threads associated to our pipeline.
    vis_source: dict[QueueConsumer]
        The list of consumers used for visualization.
    """
    def __init__(self):
        self.built = False
        self.threads = []
        self.vis_source = {}
        self.outputs = {}

    def read_vis(self):
        if not self.vis_source:
            return {}

        vis = {}
        idx = self._vis_index()
        if idx < 0:
            return {}  # This will prevent a crash while waiting for queues to be ready

        for key, consumer in self.vis_source.items():
            vis[key] = consumer.read(idx)

        return vis

    def _vis_index(self):
        return min([vis.size() for vis in self.vis_source.values()]) - 1

    def warning_wrapper(self, f):
        def wrap(*args, **kwargs):
            ret = f(*args, **kwargs)
            if ret is Pipeline._empty or isinstance(ret, Pipeline._skip):
                raise Exception('A macro block returned a Pipeline._empty or a Pipeline._skip object.')
            return ret
        return wrap

    def build_pipeline(self, pipeline_def: Graph, macros: List[List], outputs: list, vis_mode: bool, use_mp: bool) -> None:
        if self.built:
            raise Exception('The pipeline is already built')

        if vis_mode:
            outputs = []
            macros = []

        if len(pipeline_def.v()) == 0:
            return

        pipeline = copy.deepcopy(pipeline_def)
        for node in pipeline.v():
            if node.block.num_inputs() != len(pipeline.adj(node, out=False)):
                pipeline.deleteNode(node)
                log.warning('"%s" has been removed as its input were not satisfied' % node.block.name)

        ord_graph = []
        s = [node for node in pipeline.v() if node.block.num_inputs() == 0]
        while s:
            n = s.pop()
            ord_graph.append(n)
            for adj in pipeline.adj(n, out=True):
                m = adj[0]
                pipeline.deleteEdge(n, m, adj[1], adj[2])
                if not pipeline.adj(m, out=False):
                    s.append(m)

        if any([adj for adj in pipeline.adj_list]):
            raise Exception('The graph has at least one cycle')

        # We extract macro blocks by removing them from ord_graph and storing their blocks and args inside macros.
        macro_dict = {}
        if not vis_mode:
            for macro in macros:
                # They are ordered
                start_node = pipeline.get_node(macro[0])  # We won't pop the start node!
                end_node = pipeline.get_node(macro[-1])
                macro_dict[start_node] = []
                for node_hash in macro[1:-1]:
                    node = pipeline.get_node(node_hash)
                    ord_graph.remove(node)  # But we pop all the others
                    macro_dict[start_node].append((node.block, node.custom_args))
                ord_graph.remove(end_node)
                macro_dict[start_node].append(end_node)

        for node in ord_graph:
            block = node.block
            is_vis = True if block.tag == Pipeline.vis_tag else False

            # Helper to get free queues
            def get_free_out_q(node_out, out_idx):
                for i, cand in enumerate(node_out.out_queues[out_idx]):
                    q, state = cand
                    if state is False:
                        # Set output node queue state to True to prevent other from using it
                        # out_idx -> index of the out, i -> free queue, 1 -> Index of the state bool
                        node_out.out_queues[out_idx][i][1] = True
                        return q
                raise AssertionError

            # Helper to create input queues
            def get_input_queues(node):
                adj_out = pipeline_def.adj(node, out=False)
                in_q = [FakeQueue() for _ in range(block.num_inputs())]
                for node_out, out_idx, inp_idx, _ in adj_out:
                    free_q = get_free_out_q(node_out, out_idx)
                    in_q[inp_idx] = free_q
                return in_q

            # Create input and output queues
            in_q = []
            if block.num_inputs() != 0:  # If there are inputs
                in_q = get_input_queues(node)

            def chain_functions(functions):
                end_f = functions.pop()
                def wrap(*args):
                    yield from end_f(*reduce(lambda x, f: _force_tuple(next(f(*x))), functions, args))
                return wrap

            # Now that we created the inputs if we are dealing with a macro block we need
            # to use it's output block to calculate the outputs of the whole block.
            if node in macro_dict.keys():
                last_node = macro_dict[node][-1]
                last_block = last_node.block
                last_custom_args = last_node.custom_args
                start_name = block.name
                last_name = last_block.name

                funcs = [block.get_function(node.custom_args)]
                for macro_block, custom_args in macro_dict[node][:-1]:
                    # We wrap each function in a warning manager and add them to the list of functions
                    #funcs.append(macro_block.get_function(custom_args))
                    f = macro_block.get_function(custom_args)
                    if Pipeline.USE_MACRO_WARNINGS:
                        f = self.warning_wrapper(f)
                    funcs.append(f)
                funcs.append(last_block.get_function(last_custom_args))
                f = chain_functions(funcs)

                block = Block(f, False, last_block.max_queue, last_block.output_names,
                            last_block.tag, last_block.data_type, block.intercept_end, True)
                node = last_node  # Overwrite the node to the last node
                node.block.name = '%s -> %s' % (start_name, last_name)

            # Populate the output queue dictionary
            if is_vis:  # Visualization blocks have an hardcoded single queue as output
                node.out_queues = get_queue_class(use_mp, block.max_queue)
                out_q = [[node.out_queues]]
            else:
                for adj in pipeline_def.adj(node, out=True):
                    node.out_queues[adj[1]].append([get_queue_class(use_mp, block.max_queue), False])
                out_q = [[x[0] for x in out] for out in node.out_queues]

                if hash(node) in outputs:
                    log.debug('Output "%s" (%s) has been builded' % (node.name, hash(node)))
                    q = get_queue_class(use_mp, Pipeline.MAX_OUT_SIZE)
                    out_q.append(q)  # The last element of out_q is the output queue.
                    self.outputs[hash(node)] = OutputConsumer(q)

            # Create the thread
            runner = BlockRunner(block, in_q, out_q, node.custom_args, node.is_macro)
            thr = get_thread_class(use_mp, runner.run, thread_kwargs={'name': block.name})
            thr.daemon = True
            self.threads.append(thr)

            # Create the thread consumer of the visualization
            if is_vis:
                self.vis_source[str(hash(node))] = QueueConsumer(node.out_queues)
        self.built = True

    def unbuild(self):
        for i in reversed(range(len(self.threads))):
            thr = self.threads[i]
            thr.kill()
            del self.threads[i]

        for k in list(self.vis_source.keys()):
            thr = self.vis_source.pop(k)
            thr.kill()
            del thr

        self.outputs.clear()
        self.vis_source = {}
        self.built = False

    def run(self, slow: bool = False):
        if not self.built:
            raise Exception('The pipeline has not been built')

        # Start all threads
        for thr in self.threads:
            thr.slow = slow
            thr.start()
        for k, thr in self.vis_source.items():
            thr.start()

    def stop(self):
        if not self.built:
            raise Exception('The pipeline has not been built')

        for thr in self.threads:
            thr.kill()
        for k, thr in self.vis_source.items():
            thr.kill()


class QueueConsumer:
    def __init__(self, q):
        self.in_q = q
        self.out = []
        self._t = get_thread_class(self._reader)
        self._t.daemon = True

    def is_alive(self):
        return self._t.is_alive()

    def start(self):
        self._t.start()

    def join(self):
        self._t.join()

    def kill(self):
        self._t.kill()

    def _reader(self):
        while True:
            x = self.in_q.get()
            if x is StopIteration:
                self.kill()
                return
            self.out.append(x)

    def size(self):
        return len(self.out)

    def read(self, idx):
        value = None
        if len(self.out) > idx:
            value = self.out[idx]
            del self.out[:idx]
        return value


class OutputConsumer:
    def __init__(self, q):
        self.q = q
        self.run = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.run:
            raise StopIteration

        x = self.q.get()
        if x is StopIteration:
            self.run = True
            raise StopIteration
        return x


def get_queue_class(use_mp, max_size):
    base_class = mpqueues.Queue if use_mp else queue.Queue

    class CloseableQueue(base_class):
        class Closed(Exception):
            pass

        def __init__(self, maxsize):
            if isinstance(self, mpqueues.Queue):
                super(CloseableQueue, self).__init__(maxsize, ctx=mp.get_context())
                self.close_ev = mp.Event()
                self.Full = mpqueues.Full
            else:
                super(CloseableQueue, self).__init__(maxsize)
                self.close_ev = threading.Event()
                self.Full = queue.Full

        def put(self, item):
            if self.close_ev.is_set():
                raise self.Closed

            try:
                # There is a small chance that a node with full output queue is already waiting
                # to put while the queue is closed. We add a timeout to check the event again.
                super().put(item, block=True, timeout=1)
            except self.Full:
                self.put(item)

        def close(self):
            self.close_ev.set()

        def clear(self):
            while not self.empty():
                self.get()
                time.sleep(0.1)  # Leave some time to the buffer to eventually prevent a deadlock
    return CloseableQueue(max_size)


def block(f: Callable = None, max_queue: int = 10, output_names: List[str] = None,
        tag: str = 'None', data_type: str = 'raw', intercept_end: bool = False, allow_macro: bool = True):
    """
    Decorator function to tag custom blocks to be added to pipeline

    Parameters
    ----------
    f : Callable
        The function generator (or class) you are decorating (will be populated by the decorator)
    max_queue : int
        Max queue length for the output of the block
    output_names : str
        List of names for each output
    tag : str
        Tag to organize decorated blocks
    data_type : str
        For visualization blocks you have to specify the format of data you want to visualize
    intercept_end: bool
        Whether the block should intercept pipeline end and manually manage termination.
        This is a complex and advanced feature and must be implemented correctly or pipeline
        termination is not assured.
    """
    if f is None:  # Correctly manage decorator duality
        return partial(block, max_queue=max_queue, output_names=output_names, tag=tag,
                data_type=data_type, intercept_end=intercept_end, allow_macro=allow_macro)

    if isinstance(f, types.FunctionType):
        if not isgeneratorfunction(f):
            raise TypeError('The function you tagged is not a generator')
        if signature(f).parameters and list(signature(f).parameters.keys())[0] == 'self':
            raise TypeError(
                'The function you passed is a class method, we only support functions right now')
        is_class = False
    else:
        # Is a custom class so we need to process it differently (instantiate)
        assert hasattr(
            f, 'run'), 'The class %s you decorated must have a run method' % f.__name__
        if not isgeneratorfunction(f.run):
            raise TypeError('The function you tagged is not a generator')
        is_class = True

    assert data_type in Pipeline.data_type
    if tag == Pipeline.vis_tag:
        output_names = []

    Pipeline.register_block(f, is_class, max_queue, output_names,
            tag, data_type, intercept_end, allow_macro)
    return f


def _force_tuple(x):
    return x if isinstance(x, tuple) else (x,)


class BlockRunner:
    def __init__(self, block, in_q, out_q, custom_arg, is_macro):
        self.block = block
        self.in_q = in_q
        self.out_q = out_q
        self.is_macro = is_macro

        # Create full output if this is an output block
        self.full_out = None
        if isinstance(self.out_q[-1], (queue.Queue, mpqueues.Queue)):
            self.full_out = self.out_q.pop()

        self.custom_arg = custom_arg
        self.skip = False
        self.intercept = self.block.intercept_end
        self.terminate = False

        # Map the correct function from the block
        if is_macro:
            self.f = block.f
        else:
            self.f = block.get_function(self.custom_arg)

    def run(self):
        # Pipeline.empty -> The function is not ready to return anything and its output is skip
        # Pipeline._skip -> The function is still busy processing old input, do not pass another
        # If the last returned value is a Pipeline._skip object, self.skip will be True.
        if self.skip:
            # As we are skipping the input there are two possible situations:
            # 1 - The block asked for more time to end its job, this is only possible if
            # intercept_end flag is True, and we have intercept set to False
            # To allow the block to keep running in this 'overtime' state we need to keep
            # passing StopIteration to it.
            # 2 - The block just wants to skip its input, we are going to give it an empty
            fill = StopIteration if self.block.intercept_end and not self.intercept else Pipeline._empty
            x = [fill for _ in range(len(self.in_q))]  # This is going to be the input
            self.skip = False
            last_iteration = False
        else:
            x = [q.get() for q in self.in_q]  # This is going to be the input
            # If any queue returns a stop iteration we can assume the Pipeline is ending.
            last_iteration = any(v is StopIteration for v in x)

        # Here the main if else block. If we are having our 'last_iteration' aka one of our
        # inputs is a StopIteration and we cannot procede further we are gonna create a fake
        # output made of StopIteration that will be shared to our connections so that they
        # know they should terminate as well.
        # If the block has the special flag intercept_end, the node wants to intercept the
        # ending of the pipeline and we are gonna skip the 'first' last_iteration we encounter
        if last_iteration and not self.intercept:
            ret = [StopIteration for _ in range(len(self.out_q))]
            log.debug('Received a stop iteration from a queue block "%s"' % self.block.name)
            self.terminate = True
        else:
            if last_iteration:
                # If this should be the last iteration and we intercepted it because of the
                # block flag we make sure that the input queues are filled with the
                # StopIteration flags we did not process yet.
                self.intercept = False
                for q in self.in_q:
                    q.put(StopIteration)

            try:  # We are finally ready to call our function
                ret = next(self.f(*x))

                # We need to get a tuple for correct sorting of outputs, so we force it
                # But if the output has length 1 and is an actual tuple this is gonna break it
                # into parts so we convert it to a list as a workaround.
                if isinstance(ret, tuple) and self.block.num_outputs() <= 1:
                    ret = list(ret)
                ret = _force_tuple(ret)  # Force the tuple

            # If the ending was manually managed we are gonna get a nice StopIteration
            # exception but if the block is a class and does some sort of iteration that
            # throws an exception it will be seen as a RuntimeError.
            except (StopIteration, RuntimeError):
                # We need to shut down our connections, ideally this is reached by a node
                # with no inputs first and propagating a bunch of StopIteration is enough
                # we are not sure of that tho. so if the block has any input queue we are
                # gonna close them, closing a queue will make it throw an exception on put
                # the closing is done on the end of the function to be sure that the rest
                # of the function is executed.
                ret = [StopIteration for _ in range(len(self.out_q))]
                self.terminate = True  # We mark this node to be terminated
                log.debug('"%s" returned a StopIteration' % self.block.name)
            except Exception as e:  # Other exceptions are logged and a fake output is created
                ret = [Pipeline._empty for _ in range(len(self.out_q))]
                log.debug('BlockRunner block "%s" has thrown "%s"' % (self.block.name, e))

        # If the function returned a _skip object we need to process it correctly.
        # We read the data stored inside of it and mark the node to skip next input
        if len(ret) == 1 and isinstance(ret[0], Pipeline._skip):
            self.skip = True
            ret = tuple(re.x for re in ret)

        # Fill the output queues (the one tagged using pipeline.add_output) with the values
        # returned by the function (squeezed if possible) if the values returned are usable
        # aka at least one of them is not an _empty objects (ideally none of them)
        if self.full_out:
            if not all([re is Pipeline._empty for re in ret]):
                self.full_out.put(ret if len(ret) > 1 else ret[0])

        # Some of the output queues may be closed, but we still want to serve the open ones!
        # If all of them are closed we have no purpose and can terminate.
        all_died = False  # Set to false by default (as we can have no outputs)
        if any(out for out in self.out_q):
            all_died = True
            for i, out in enumerate(self.out_q):
                for q in out:
                    v = ret[i]
                    if v is not Pipeline._empty:
                        try:
                            q.put(v)
                            all_died = False  # If at least one is open keep running
                        except q.Closed:  # On closed queue exception
                            q.clear()
                            log.debug('"%s" found queue closed' % self.block.name)

        if self.terminate or all_died:  # If all outputs died or instructed to terminate
            if self.full_out:
                self.full_out.put(StopIteration)
            for q in self.in_q:
                q.close()
            raise StopIteration


class FakeQueue:
    def get(self):
        raise Exception('This is a fake queue and should be replaced')


def get_thread_class(use_mp, *args, **kwargs):
    base_class = mp.Process if use_mp else threading.Thread

    class TerminableThread(base_class):
        def __init__(self, f, thread_args=(), thread_kwargs={}, *args, **kwargs):
            super(TerminableThread, self).__init__(*thread_args, **thread_kwargs)
            self.target = lambda: f(*args, **kwargs)
            self.slow = False
            if isinstance(self, mp.Process):
                self._killer = mp.Event()
            else:
                self._killer = threading.Event()

        def __del__(self):
            log.debug('Deleted Thread Successfully')

        def kill(self):
            self._killer.set()

        def _killed(self):
            return self._killer.is_set()

        def run(self):
            while True:
                if self._killed():
                    return

                try:
                    self.target()
                except StopIteration:
                    self.kill()

                if self.slow:
                    time.sleep(0.5)

    return TerminableThread(*args, **kwargs)
