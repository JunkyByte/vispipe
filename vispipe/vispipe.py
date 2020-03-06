from vispipe.node import Node, Block
from vispipe.graph import Graph
from functools import partial
from inspect import signature, isgeneratorfunction
from typing import Callable
from threading import Thread, Event
from queue import Queue
from typing import List, Union
import types
import copy
import pickle
import time
MAXSIZE = 100

# TODO: Macro blocks? to execute multiple nodes subsequently, while it's impractical to run them in a faster way, I suppose that just creating a way to define them can be convenient.
# TODO: Blocks with inputs undefined? Like tuple together all the inputs, how to?
# TODO: When inputting custom args from visualization set them as current text as they are not modifiable
# TODO: If more than one output on delete not cleaning correctly


class Pipeline:
    """
    Pipeline class that is used to represent pipelines.

    Attributes
    ----------
    pipeline: :class:`.vispipe.graph.Graph`
    runner: :class:`.PipelineRunner`

    """
    class _skip_class:
        def __call__(self, x):
            self.x = x
            return self

    #: Yield this to specify that a block is not ready to create a new output.
    _empty = object()

    #: Yield this to specify that a block is busy processing old input, on next call no new data will be passed.
    _skip = _skip_class()

    # TODO: Plot? Should be converted internally to an image
    data_type = ['raw', 'image', 'plot']  #: Supported data_type for visualization blocks.
    vis_tag = 'vis'  #: The tag used for visualization blocks.

    _blocks = {}

    @staticmethod
    def register_block(func: Callable, is_class: bool, max_queue: int, output_names: List[str] = None,
            tag: str = 'None', data_type: str = 'raw') -> None:
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
        """
        block = Block(func, is_class, max_queue, output_names, tag, data_type)
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



    def __init__(self):
        self.pipeline = Graph(MAXSIZE)
        self.runner = PipelineRunner()

    def nodes(self) -> List[Node]:
        """
        Returns the list of :class:`.Node` that are part of the pipeline.

        Returns
        -------
        List[Node]:
            The list of nodes.
        """
        return self.pipeline.v()

    def connections(self, node_hash: int, out=None) -> List[tuple]:
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
        List[tuple]:
            The list of connections.
        """
        node = self.get_node(node_hash)
        return self.pipeline.adj(node, out)

    def add_node(self, block_name: str, **kwargs) -> int:
        """
        Adds a new node to the pipeline.

        Parameters
        ----------
        block_name : str
            The name of the block you want to add.
        **kwargs
            Used to specify custom arguments for the block. 

        Returns
        -------
        int:
            The hash of the created node
        """
        node = Node(Pipeline._blocks[block_name], **kwargs)
        return self.pipeline.insertNode(node)

    def get_node(self, node_hash: int):
        return self.pipeline.get_node(node_hash)

    def remove_node(self, node_hash: int):
        node = self.get_node(node_hash)
        self.pipeline.deleteNode(node)

    def clear_pipeline(self):
        self.pipeline.resetGraph()
        self.unbuild()

    def add_conn(self, from_hash: int, out_index: int, to_hash: int, inp_index: int):
        from_node = self.get_node(from_hash)
        to_node = self.get_node(to_hash)
        self.pipeline.insertEdge(from_node, to_node, out_index, inp_index)

    def set_custom_arg(self, node_hash: str, key: str, value):
        node = self.get_node(node_hash)
        arg_type = node.block.custom_args_type[key]
        if arg_type is bool and isinstance(value, str):
            if value.lower() == 'true':
                node.custom_args[key] = True
            elif value.lower() == 'false':
                node.custom_args[key] = False
        else:
            node.custom_args[key] = arg_type(value)

    def build(self) -> None:
        self.runner.build_pipeline(self.pipeline)

    def unbuild(self) -> None:
        for node in self.pipeline.v():
            node.clear_out_queues()
        self.runner.unbuild()

    def run(self, slow=False) -> None:
        self.runner.run(slow)

    def stop(self) -> None:
        self.runner.stop()

    def save(self, path, vis_data={}) -> None:
        with open(path, 'wb') as f:
            pickle.dump((self.pipeline, vis_data), f)

    def load(self, path) -> dict:
        self.clear_pipeline()
        with open(path, 'rb') as f:
            self.pipeline, vis_data = pickle.load(f)
        return self.pipeline, vis_data


class PipelineRunner:
    """
    The class used to run pipelines.

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

    def read_vis(self):
        """
        Reads the current data passing from all the visualization blocks.

        Returns
        -------
        dict
            The current visualization data.
        """
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

    def build_pipeline(self, pipeline_def: Graph) -> None:
        """
        Builds the pipeline and makes it ready to be run.
        It creates the list of associated :class:`.TerminableThread` and :class:`.QueueConsumer`
        and binds them together using :obj:`Queue`

        To build the pipeline graph we follow the `Kahn's Algorithm for topologic sorting <https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm>`_
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

        Parameters
        ----------
        pipeline_def : Graph
            The pipeline we want to build.
        """
        assert not self.built
        if len(pipeline_def.v()) == 0:
            return

        pipeline = copy.deepcopy(pipeline_def)
        for node in pipeline.v():
            if node.block.num_inputs() != len(pipeline.adj(node, out=False)):
                pipeline.deleteNode(node)
                print('%s has been removed as its input were not satisfied' % node.block.name)

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

        for node in ord_graph:
            block = node.block
            is_vis = True if block.tag == Pipeline.vis_tag else False

            # Helper to get free queues
            def get_free_out_q(node_out, out_idx):
                for i, cand in enumerate(node_out.out_queues[out_idx]):
                    q, state = cand
                    if state is False:
                        node_out.out_queues[out_idx][i][1] = True
                        return q
                raise AssertionError

            # Helper to create input queues
            def get_input_queues(node):
                adj_out = pipeline_def.adj(node, out=False)
                in_q = [FakeQueue() for _ in range(node.block.num_inputs())]
                for node_out, out_idx, inp_idx, _ in adj_out:
                    free_q = get_free_out_q(node_out, out_idx)
                    in_q[inp_idx] = free_q
                return in_q

            # Create input and output queues
            in_q = []
            if block.num_inputs() != 0:  # If there are inputs
                in_q = get_input_queues(node)

            # Populate the output queue dictionary
            if is_vis:  # Visualization blocks have an hardcoded single queue as output
                node.out_queues = Queue(node.block.max_queue)
                out_q = [[node.out_queues]]
            else:
                for adj in pipeline_def.adj(node, out=True):
                    node.out_queues[adj[1]].append([Queue(node.block.max_queue), False])
                out_q = [[x[0] for x in out] for out in node.out_queues]

            # Create the thread
            runner = BlockRunner(node.block, in_q, out_q, node.custom_args)
            thr = TerminableThread(runner.run)
            thr.daemon = True
            self.threads.append(thr)

            # Create the thread consumer of the visualization
            if is_vis:
                self.vis_source[str(hash(node))] = QueueConsumer(node.out_queues)
        self.built = True

    def unbuild(self):
        """
        Unbuild the pipeline by destroying the threads and consumers.
        """
        for i in reversed(range(len(self.threads))):
            thr = self.threads[i]
            if thr.is_alive():
                thr.kill()
            del self.threads[i]

        for k in list(self.vis_source.keys()):
            thr = self.vis_source.pop(k)
            if thr.is_alive():
                thr.kill()
            del thr

        self.vis_source = {}
        self.built = False

    def run(self, slow: bool = False):
        """
        Run the pipeline (it has to be built already). 

        Parameters
        ----------
        slow : bool
            Whether to run the pipeline in slow mode, useful for visualization and debugging.
        """
        if not self.built:
            raise Exception('The pipeline has not been built')

        # Start all threads
        for thr in self.threads:
            thr.slow = slow

            if thr.is_stopped():
                thr.resume()
            else:
                thr.start()
        for k, thr in self.vis_source.items():
            if thr.is_stopped():
                thr.resume()
            else:
                thr.start()

    def stop(self):
        """
        Stops the pipeline.
        """
        if not self.built:
            raise Exception('The pipeline has not been built')

        for thr in self.threads:
            thr.stop()
        for k, thr in self.vis_source.items():
            thr.stop()


class QueueConsumer:
    def __init__(self, q):
        self.in_q = q
        self.out = []
        self._t = TerminableThread(self._reader)
        self._t.daemon = True

    def is_alive(self):
        return self._t.is_alive()

    def start(self):
        self._t.start()

    def stop(self):
        self._t.stop()

    def is_stopped(self):
        return self._t.is_stopped()

    def join(self):
        self._t.join()

    def kill(self):
        self._t.kill()

    def resume(self):
        self._t.resume()

    def _reader(self):
        while True:
            x = self.in_q.get()
            self.out.append(x)

    def size(self):
        return len(self.out)

    def read(self, idx):
        value = None
        if len(self.out) > idx:
            value = self.out[idx]
            del self.out[:idx]
        return value


def block(f: Callable = None, max_queue: int = 2, output_names: str = None, tag: str = 'None', data_type: str = 'raw'):
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
    """
    if f is None:  # Correctly manage decorator duality
        return partial(block, max_queue=max_queue, output_names=output_names, tag=tag, data_type=data_type)

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

    Pipeline.register_block(f, is_class, max_queue,
                            output_names, tag, data_type)
    return f


def force_tuple(x):
    return x if isinstance(x, tuple) else (x,)


class BlockRunner:
    def __init__(self, node, in_q, out_q, custom_arg):
        self.node = node
        self.in_q = in_q
        self.out_q = out_q
        self.custom_arg = custom_arg
        self.skip = False

        if node.is_class:
            self.f = node.f(**self.custom_arg).run
        else:
            self.f = partial(node.f, **self.custom_arg)

    def run(self):
        # Pipeline.empty -> The function is not ready to return anything and you should skip its output
        # Pipeline._skip -> The function is still busy processing old input, do not pass another
        if self.skip:
            x = [Pipeline._empty for _ in range(len(self.in_q))]
            self.skip = False
        else:
            x = [q.get() for q in self.in_q]

        try:
            ret = next(self.f(*x))
            if isinstance(ret, tuple) and self.node.num_outputs() <= 1:
                ret = list(ret)
            ret = force_tuple(ret)

        except Exception as e:
            ret = [Pipeline._empty for _ in range(len(self.out_q))]
            print('BlockRunner node: %s has thrown: %s' % (self.node.name, e))

        if len(ret) == 1 and isinstance(ret[0], Pipeline._skip_class):
            self.skip = True
            ret = [re.x for re in ret]
        for i, out in enumerate(self.out_q):
            for q in out:
                v = ret[i]
                if v is not Pipeline._empty:
                    q.put(v)


class FakeQueue:
    def get(self):
        return []


class TerminableThread(Thread):
    def __init__(self, f, thread_args=(), thread_kwargs={}, *args, **kwargs):
        super(TerminableThread, self).__init__(*thread_args, **thread_kwargs)
        self._killer = Event()
        self._pause = Event()
        self.name = f.__name__
        self.target = lambda: f(*args, **kwargs)
        self.slow = False

    def __del__(self):
        print('Deleted Thread Successfully')  # TODO: Remove me

    def kill(self):
        self._killer.set()

    def stop(self):
        self._pause.set()

    def resume(self):
        self._pause.clear()

    def is_stopped(self):
        return self._pause.is_set()

    def _killed(self):
        return self._killer.isSet()

    def _stopped(self):
        return self._pause.isSet()

    def run(self):
        while True:
            if self._killed():
                return

            if self._stopped():
                time.sleep(0.5)
                continue

            self.target()
            if self.slow:
                time.sleep(1)

