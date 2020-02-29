from vispipe.node import Node
from vispipe.graph import Graph
from functools import partial
from inspect import signature, isgeneratorfunction, _empty
from typing import List, Callable
from threading import Thread, Event
from queue import Queue
import types
import copy
import pickle
import time
MAXSIZE = 100


class Pipeline:
    """
    The Pipeline object you will interact with.
    You should not instantiate your own Pipeline, use the one provided by vispipe.
    """
    class _skip_class:
        def __call__(self, x):
            self.x = x
            return self

    _empty = object()
    _skip = _skip_class()

    # TODO: Plot? Should be converted internally to an image
    data_type = ['raw', 'image', 'plot']
    vis_tag = 'vis'

    def __init__(self):
        self._blocks = {}
        self.pipeline = Graph(MAXSIZE)
        self.runner = PipelineRunner()

    def get_blocks(self, serializable=False):
        blocks = []
        for block in self._blocks.values():
            if serializable:
                block = block.serialize()
            else:
                block = dict(block)
            blocks.append(block)
        return blocks

    def register_block(self, func: Callable, is_class: bool, max_queue: int, output_names=None,
            tag: str = 'None', data_type: str = 'raw') -> None:
        block = Block(func, is_class, max_queue, output_names, tag, data_type)
        assert block.name not in self._blocks.keys(), 'The name %s is already registered as a pipeline block' % block.name
        self._blocks[block.name] = block

    def nodes(self):
        return self.pipeline.v()

    def connections(self, node_hash: int, out=None):
        node = self.get_node(node_hash)
        return self.pipeline.adj(node, out)

    def add_node(self, block, **kwargs) -> Node:
        node = Node(block, **kwargs)
        return self.pipeline.insertNode(node)

    def get_node(self, node_hash: int):
        return self.pipeline.get_node(node_hash)

    def remove_node(self, node_hash: int):
        node = self.get_node(node_hash)
        self.pipeline.deleteNode(node)

    def clear_pipeline(self):
        self.pipeline.resetGraph()
        self.runner.unbuild()

    def add_conn(self, from_hash: int, out_index: int, to_hash: int, inp_index: int):
        from_node = self.get_node(from_hash)
        to_node = self.get_node(to_hash)
        self.pipeline.insertEdge(from_node, to_node, out_index, inp_index)

    def set_custom_arg(self, node_hash: str, key: str, value):
        node = self.get_node(node_hash)
        arg_type = node.block.custom_args_type[key]
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
        with open(path, 'rb') as f:
            self.pipeline, vis_data = pickle.load(f)
        return self.pipeline, vis_data


class PipelineRunner:
    def __init__(self):
        self.built = False
        self.threads = []
        self.vis_source = {}

    def read_vis(self):
        vis = {}
        idx = self._vis_index()
        if idx < 0:
            return {}  # This will prevent a crash while waiting for queues to be ready

        for key, consumer in self.vis_source.items():
            vis[key] = consumer.read(idx)

        return vis

    def _vis_index(self):
        return min([vis.size() for vis in self.vis_source.values()]) - 1

    def build_pipeline(self, pipeline_def):
        assert not self.built
        if len(pipeline_def.v()) == 0:
            return

        """
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
        """
        pipeline = copy.deepcopy(pipeline_def)
        for node in pipeline.v():  # TODO: This can be checked inside the algo
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

    def run(self, slow=False):
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


class Block:
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
        self.input_args = dict([(k, val) for k, val in args if val == _empty])
        self.custom_args = dict([(k, val) for k, val in args if val != _empty])
        self.custom_args_type = dict([(k, type(val)) for k, val in self.custom_args.items()])
        self.max_queue = max_queue
        self.output_names = output_names if output_names is not None else ['y']  # Not a mistake, it has to catch empty list

    def num_inputs(self):
        return len(self.input_args)

    def num_outputs(self):
        return len(self.output_names)

    def serialize(self):
        x = dict(self)
        del x['f']
        # TODO: Change me to a custom value, using None can cause problems
        # TODO: Support np array by casting to nested lists
        x['input_args'] = dict([(k, v if v != _empty else None)
                                for k, v in x['input_args'].items()])
        x['custom_args_type'] = dict([(k, str(v.__name__)) for k, v in x['custom_args_type'].items()])
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

    assert data_type in pipeline.data_type
    if tag == pipeline.vis_tag:
        output_names = []

    pipeline.register_block(f, is_class, max_queue,
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

        ret = force_tuple(next(self.f(*x)))

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
                time.sleep(0.25)
                continue

            self.target()
            if self.slow:
                time.sleep(0.5)


pipeline = Pipeline()
