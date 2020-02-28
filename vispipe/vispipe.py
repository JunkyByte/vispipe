from functools import partial
from inspect import signature, isgeneratorfunction, _empty
from typing import List, Callable
from threading import Thread, Event
from queue import Queue, Empty
import os
import types
import copy
import pickle
import numpy as np
import time
from vispipe.Graph import Graph
MAXSIZE = 100
assert np.log10(MAXSIZE) == int(np.log10(MAXSIZE))


class Pipeline:
    _empty = object()
    # TODO: Plot? Should be converted internally to an image
    data_type = ['raw', 'image', 'plot']
    vis_tag = 'vis'

    def __init__(self):
        self._blocks = {}
        self.pipeline = Graph(MAXSIZE)
        self.runner = PipelineRunner()
        self.nodes = []

    def get_blocks(self, serializable=False):
        blocks = []
        for block in self._blocks.values():
            if serializable:
                block = block.serialize()
            else:
                block = dict(block)
            blocks.append(block)
        return blocks

    def register_block(self, func: Callable, is_class: bool,max_queue: int, output_names = None, tag: str = 'None', data_type: str = 'raw'):
        block = Block(func, is_class, max_queue, output_names, tag, data_type)
        assert block.name not in self._blocks.keys(), 'The name %s is already registered as a pipeline block' % block.name
        self._blocks[block.name] = block

    def add_node(self, block, **kwargs):
        node = Node(block, **kwargs)
        self.nodes.append(node)
        return self.pipeline.insertNode(node)

    def remove_node(self, node_hash):
        node = self.pipeline.get_node(node_hash)
        self.pipeline.deleteNode(node)

    def clear_pipeline(self):
        self.pipeline.resetGraph()
        self.runner.unbuild()

    def add_conn(self, from_hash, out_index, to_hash, inp_index):
        from_node = self.pipeline.get_node(from_hash)
        to_node = self.pipeline.get_node(to_hash)
        self.pipeline.insertEdge(from_node, to_node, out_index, inp_index)

    def build(self) -> None:
        self.runner.build_pipeline(self.pipeline)

    def unbuild(self) -> None:
        for node in self.nodes:
            node.out_queues = []
        self.runner.unbuild()

    def run(self) -> None:
        self.runner.run()

    def stop(self) -> None:
        self.runner.stop()

    def save(self, path) -> None:
        with open(os.path.join(path, 'pipeline.pickle'), 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load(self, path) -> None:
        with open(os.path.join(path, 'pipeline.pickle'), 'rb') as f:
            self.pipeline = pickle.load(f)


class PipelineRunner:
    def __init__(self):
        self.built = False
        self.threads = []
        self.out_queues = {}
        self.vis_source = {}

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
                for node_out, out_idx, inp_idx, _  in adj_out:
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

    def run(self):
        if not self.built:
            raise Exception('The pipeline has not been built')

        # Start all threads
        for thr in self.threads:
            if thr.is_stopped():
                thr.resume()
            else:
                thr.start()
        for k, thr in self.vis_source.items():
            if thr.is_stopped():
                thr.resume()
            else:
                thr.start()

        # Join all threads TODO
        #for thr in self.threads:
        #    thr.join()
        #for k, thr in self.vis_source.items():
        #    thr.join()

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
        self.out_q = Queue()
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

    # read from queue as soon as possible, keeping only most recent result
    def _reader(self):
        while True:
            x = self.in_q.get()
            if not self.out_q.empty():
                try:
                    self.out_q.get_nowait()
                except Empty:
                    pass
            self.out_q.put(x)

    def read(self):
        return self.out_q.get()


# class PipelineGraph:
#     def __init__(self):
#         self.matrix = np.empty(
#             (MAXSIZE, MAXSIZE), dtype=object)  # Adjacency matrix
#         self.nodes = np.empty((MAXSIZE,), dtype=Block)
#         self.custom_args = np.empty((MAXSIZE,), dtype=dict)
#
#         self.ids = {}  # Map from hash to assigned ids
#         self.instances = {}  # Free ids for instances of same block
#         self.free_idx = set([i for i in range(MAXSIZE)])  # Free ids for blocks
#
#     def index_to_id(self, block, index):
#         for k, v in self.ids.items():
#             if v == index:
#                 hash_block = self.get_hash(block, index=0)[1]
#                 return k - hash_block * MAXSIZE
#         assert False, 'The index has not been found'
#
#     def get_hash(self, block, index=None):
#         hash_block = hash(block)
#         if hash_block not in self.instances.keys() and index is None:
#             self.instances[hash_block] = set([i for i in range(MAXSIZE)])
#
#         if index is None:
#             index = self.instances[hash_block].pop()
#         hash_index = hash_block * MAXSIZE + index
#         return (hash_index, hash_block)
#
#     def hash_to_instance(self, hash_index, hash_block):
#         return hash_index - hash_block * MAXSIZE
#
#     def register_node(self, block, id):
#         hash_index, hash_block = self.get_hash(block)
#         assert hash_index not in self.ids.keys()
#         self.ids[hash_index] = id
#         return self.hash_to_instance(hash_index, hash_block)
#
#     def add_node(self, block, **kwargs):
#         id = self.free_idx.pop()
#         index = self.register_node(block, id)  # Register block in the matrix
#         num_outputs = block.num_outputs()
#         self.nodes[id] = block  # Store the node instance
#         self.custom_args[id] = kwargs
#         out = np.array([0 for _ in range(num_outputs)])
#         for i in range(MAXSIZE):
#             self.matrix[id, i] = out
#         return index
#
#     def remove_node(self, block, index):
#         hash_index, hash_block = self.get_hash(block, index=index)
#         id = self.ids.pop(hash_index)
#         self.nodes[id] = None  # unassign the node instance
#         self.custom_args[id] = None
#         self.matrix[id, :] = None  # delete the node connections
#         self.free_idx.add(id)
#         self.instances[hash_block].add(self.hash_to_instance(hash_index, hash_block))
#
#     def clear_pipeline(self):
#         self.matrix[:, :] = None
#         self.nodes[:] = None
#         self.custom_args[:] = None
#         self.free_idx = set([i for i in range(MAXSIZE)])
#         self.ids = {}
#
#     def add_connection(self, from_block, from_idx, out_idx, to_block, to_idx, inp_idx):
#         assert inp_idx < to_block.num_inputs()
#         assert out_idx < from_block.num_outputs()
#         assert from_block.tag != Pipeline.vis_tag  # Prevent connections from vis blocks
#         hash_from, hash_from_block = self.get_hash(from_block, from_idx)
#         hash_to, hash_to_block = self.get_hash(to_block, to_idx)
#         assert hash_from != hash_to
#
#         from_id = self.ids[hash_from]
#         to_id = self.ids[hash_to]
#
#        self.matrix[from_id][to_id] = np.array([inp_idx + 1])


class Node:
    def __init__(self, node_block, **kwargs):
        self.block = node_block
        self.custom_args = kwargs
        self.out_queues = []
        self._hash = None
        for _ in range(self.block.num_outputs()):
            self.out_queues.append([])

    def __hash__(self):
        if self._hash is None:
            return id(self)
        else:
            return self._hash


class Block:
    def __init__(self, f: Callable, is_class: bool, max_queue: int, output_names: List[str], tag: str, data_type: str):
        self.f = f
        self.name = f.__name__
        self.is_class = is_class
        self.tag = tag
        self.data_type = data_type
        if self.is_class:
            input_args = dict(signature(self.f.run).parameters)
            del input_args['self']
        else:
            input_args = dict(signature(self.f).parameters)
        args = [(k, val.default) for k, val in input_args.items()]
        self.input_args = dict([(k, val) for k, val in args if val == _empty])
        self.custom_args = dict([(k, val) for k, val in args if val != _empty])
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
        x['input_args'] = dict([(k, v if v != _empty else None)
                                for k, v in x['input_args'].items()])
        return x

    def __iter__(self):
        yield 'f', self.f
        yield 'name', self.name
        yield 'input_args', self.input_args
        yield 'custom_args', self.custom_args
        yield 'max_queue', self.max_queue
        yield 'output_names', self.output_names
        yield 'tag', self.tag
        yield 'data_type', self.data_type


def block(f=None, max_queue=2, output_names=None, tag='None', data_type='raw'):
    """
    Decorator function to tag custom blocks to be added to pipeline
    :param f: The function to tag (as a decorator it will be automatically passed)
    :param max_queue: Max queue length for the output of the block
    :param output_names: List of names for each output
    :param tag: Tag to organize decorated blocks
    :return:
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
    if tag == pipeline.vis_tag:  # TODO: Is this a good idea?
        output_names = []

    pipeline.register_block(f, is_class, max_queue,
                            output_names, tag, data_type)
    return f


class BlockRunner:
    def __init__(self, node, in_q, out_q, custom_arg):
        self.node = node
        self.in_q = in_q
        self.out_q = out_q
        self.custom_arg = custom_arg

        if node.is_class:
            self.f = node.f().run
        else:
            self.f = node.f

    def run(self):
        x = [q.get() for q in self.in_q]
        ret = list(self.f(*x, **self.custom_arg))
        for i, out in enumerate(self.out_q):
            for q in out:
                v = ret[i]
                if v is not pipeline._empty:
                    q.put(v)


class FakeQueue:
    def get(self):
        return []


class TerminableThread(Thread):
    # Thread class with a _stop() method.
    # The thread itself has to check
    # regularly for the stopped() condition.
    def __init__(self, f, thread_args=(), thread_kwargs={}, *args, **kwargs):
        super(TerminableThread, self).__init__(*thread_args, **thread_kwargs)
        self._killer = Event()
        self._pause = Event()
        self.name = f.__name__
        self.target = lambda: f(*args, **kwargs)

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


pipeline = Pipeline()
