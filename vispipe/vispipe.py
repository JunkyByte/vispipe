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

    def register_block(self, func: Callable, is_class: bool, max_queue: int, output_names=None, tag: str = 'None', data_type: str = 'raw') -> None:
        block = Block(func, is_class, max_queue, output_names, tag, data_type)
        assert block.name not in self._blocks.keys(), 'The name %s is already registered as a pipeline block' % block.name
        self._blocks[block.name] = block

    def add_node(self, block, **kwargs):
        #return self.pipeline.add_node(block, **kwargs)
        node = Node(block, **kwargs)
        self.nodes.append(node)
        return self.pipeline.insertNode(node)

    def remove_node(self, node_hash):
        node = self.pipeline.get_node(node_hash)
        self.pipeline.deleteNode(node)

    def clear_pipeline(self):
        self.pipeline.resetGraph()
        self.runner.unbuild()  # todo refactor

    def add_conn(self, from_hash, out_index, to_hash, inp_index):
        from_node = self.pipeline.get_node(from_hash)
        to_node = self.pipeline.get_node(to_hash)
        self.pipeline.insertEdge(from_node, to_node, out_index, inp_index)

    def build(self) -> None:
        self.runner.build_pipeline(self.pipeline)

    def unbuild(self) -> None:
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

    def build_pipeline(self, pipeline):
        assert not self.built
        if len(pipeline.v()) == 0:
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
        print(pipeline.__dict__)

        nodes = pipeline.v()
        adj_copy = [copy.copy(el) for el in pipeline.adj_list]
        adj_in = []
        adj_out = []
        for adj in adj_copy:
            adj_out.append([adj_out for adj_out in adj if adj_out[-1]])
            adj_in.append([adj_out for adj_out in adj if not adj_out[-1]])

        l = []
        s = [node for node in nodes if node.block.num_inputs() == 0]
        while s:
            n = s.pop()
            id_n = pipeline.lookup(n)
            l.append(n)
            for adj in adj_out[id_n]:
                m = adj[0]
                id_m = pipeline.lookup(m)
                adj_out[id_n].remove(adj)  # Remove the adj
                adj_in[id_m].remove((n, adj[1], adj[2], False))  # Remove the opposite adj
                if not adj_in[id_m]:
                    s.append(m)

        if any([adj for adj in adj_out]):
            raise Exception('The graph has at least one cycle')
        else:
            nodes = l

        for node in l:
            block = node.block
            is_vis = True if block.tag == Pipeline.vis_tag else False
            if is_vis:
                data_type = block.data_type

            custom_arg = node.custom_args
            print(pipeline.adj(node, out=True))
            out_conn = np.array(list(nodes_conn[idx]))
            in_conn = nodes_conn[:, idx]
            # out_conn_total = np.count_nonzero(out_conn)
            out_conn_split = np.count_nonzero(out_conn, axis=0)
            in_conn_count = np.count_nonzero(np.concatenate(in_conn))

            # If inputs are not satisfied we trash the node and continue the processing
            if in_conn_count < node.num_inputs():
                trash.append(idx)
                to_process.pop(i)
                continue

            # Helper to get free queues
            def get_free_out_q(conn_id, out_idx):
                for i, cand in enumerate(self.out_queues[conn_id][out_idx]):
                    q, state = cand
                    if state is False:
                        self.out_queues[conn_id][out_idx][i][1] = True
                        return q
                raise AssertionError

            # Helper to create input queues
            def get_input_queues(conn):
                in_q = [FakeQueue() for _ in range(node.num_inputs())]
                for conn_id, value in conn:
                    for out_idx, inp_idx in enumerate(value):
                        if inp_idx == 0:
                            continue

                        free_q = get_free_out_q(conn_id, out_idx)
                        in_q[inp_idx - 1] = free_q
                return in_q

            # Create input and output queues
            in_q = []
            if node.num_inputs() != 0:  # If there are inputs
                conn = [(k, value)
                        for k, value in enumerate(in_conn) if value.any() != 0]
                try:  # Try to build dependencies (they can not exist at this time)
                    in_q = get_input_queues(conn)
                except KeyError:
                    i = i + 1 % len(to_process)
                    continue

            # Populate the output queue dictionary
            if is_vis:  # Visualization blocks have an hardcoded single queue as output
                instance_idx = pipeline.index_to_id(node, used_ids[idx])
                q = Queue(node.max_queue)
                out_q = [[q]]
            else:
                self.out_queues[idx] = [[[Queue(node.max_queue), False] for _ in range(out)]
                                        for out in out_conn_split]
                out_q = [[x[0] for x in out] for out in self.out_queues[idx]]

            # Create the thread
            runner = BlockRunner(node, in_q, out_q, custom_arg)
            thr = TerminableThread(runner.run)
            thr.daemon = True
            self.threads.append(thr)

            # Create the thread consumer of the visualization
            if is_vis:
                self.vis_source[(node, instance_idx)] = QueueConsumer(q)

            to_process.pop(i)
            i = 0  # If we successfully processed a node we go back to the highest priority
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
        
        self.out_queues = {}
        self.vis = {}
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
        for _ in range(self.block.num_outputs()):
            self.out_queues.append([])

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
