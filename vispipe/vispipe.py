import numpy as np
from functools import partial
from inspect import signature, isgeneratorfunction, _empty
from typing import List, Callable
from threading import Thread, Event
from queue import Queue
import types

MAXSIZE = 100
assert np.log10(MAXSIZE) == int(np.log10(MAXSIZE))

class Pipeline:
    def __init__(self):
        self._blocks = {}
        self.pipeline = PipelineGraph()
        self.runner = PipelineRunner()

    def register_block(self, f : Callable, is_class: bool, max_queue : int, output_names=None) -> None:
        block = Block(f, is_class, max_queue, output_names)
        assert block.name not in self._blocks.keys(), 'The name %s is already registered as a pipeline block' % block.name
        self._blocks[block.name] = block

    def add_node(self, block, **kwargs):
        return self.pipeline.add_node(block, **kwargs)

    def remove_node(self, block, index):
        self.pipeline.remove_node(block, index)

    def add_conn(self, from_block, from_idx, out_idx, to_block, to_idx, inp_idx):
        self.pipeline.add_connection(from_block, from_idx, out_idx, to_block, to_idx, inp_idx)

    def build(self) -> None:
        self.runner.build_pipeline(self.pipeline)

    def run(self) -> None:
        self.runner.run()

class PipelineRunner:
    def __init__(self):
        self.built = False
        self.threads = []
        self.out_queues = {}
        self.out_queues_state = []

    def build_pipeline(self, pipeline):
        # Collect each block arguments, create connections between each other using separated threads
        # use queues to pass data between the threads.
        used_ids = np.array(list(pipeline.ids.values()))
        used_ids.sort()
        nodes = pipeline.nodes[used_ids]
        custom_args = pipeline.custom_args[used_ids]
        nodes_conn = np.array([x[used_ids] for x in pipeline.matrix[used_ids]])
        gain = np.array([n.num_outputs() / (n.num_inputs() + 1e-16) for n in nodes])
        to_process = list(np.arange(len(used_ids))[(-gain).argsort()])
        trash = []

        i = 0
        while to_process != []:
            idx = to_process[i]

            node = nodes[idx]
            custom_arg = custom_args[idx]
            out_conn = np.array(list(nodes_conn[idx]))
            in_conn = nodes_conn[:, idx]
            out_conn_total = np.count_nonzero(out_conn)
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
                    if state == False:
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
                conn = [(k, value) for k, value in enumerate(in_conn) if value.any() != 0]
                try:  # Try to build dependencies (they can not exist at this time)
                    in_q = get_input_queues(conn)
                except KeyError:
                    i = i + 1 % len(to_process)
                    continue

            # Populate the output queue dictionary
            self.out_queues[idx] = [[[Queue(node.max_queue), False] for _ in range(out)] for out in out_conn_split]
            out_q = [[x[0] for x in out] for out in self.out_queues[idx]]

            # Create the thread
            runner = BlockRunner(node, in_q, out_q, custom_arg)
            thr = TerminableThread(runner.run)
            thr.daemon = True
            self.threads.append(thr)
            to_process.pop(i)
            i = 0  # If we successfully processed a node we go back to the highest priority
        self.built = True

    def run(self):
        if self.built:
            for thr in self.threads:
                thr.start()
            for thr in self.threads:
                thr.join()
        else:
            raise Exception('The pipeline has not been built')

class PipelineGraph:
    def __init__(self):
        self.matrix = np.empty((MAXSIZE, MAXSIZE), dtype=object)  # Adjacency matrix
        self.nodes = np.empty((MAXSIZE,), dtype=Block)
        self.custom_args = np.empty((MAXSIZE,), dtype=dict)

        self.ids = {}  # Map from hash to assigned ids
        self.instances = {}  # Free ids for instances of same block
        self.free_idx = set([i for i in range(MAXSIZE)])  # Free ids for blocks

    def get_hash(self, block, index=None):
        hash_block = hash(block)
        if not hash_block in self.instances.keys() and index is None:
            self.instances[hash_block] = set([i for i in range(MAXSIZE)])

        if index is None:
            index = self.instances[hash_block].pop()
        hash_index = hash_block * MAXSIZE + index
        return (hash_index, hash_block)

    def hash_to_instance(self, hash_index, hash_block):
        return hash_index - hash_block * MAXSIZE

    def register_node(self, block, id):
        hash_index, hash_block = self.get_hash(block)
        assert hash_index not in self.ids.keys()
        self.ids[hash_index] = id
        return self.hash_to_instance(hash_index, hash_block)

    def add_node(self, block, **kwargs):
        id = self.free_idx.pop()
        index = self.register_node(block, id)  # Register block in the matrix
        num_outputs = block.num_outputs()
        self.nodes[id] = block  # Store the node instance
        self.custom_args[id] = kwargs
        out = np.array([0 for _ in range(num_outputs)])
        for i in range(MAXSIZE): self.matrix[id, i] = out
        return index

    def remove_node(self, block, index):
        hash_index, hash_block = self.get_hash(block, index=index)
        id = self.ids.pop(hash_index)
        self.nodes[id] = None  # unassign the node instance
        self.custom_args[id] = None
        self.matrix[id, :] = None  # delete the node connections
        self.instances[hash_block].add(self.hash_to_instance(hash_index, hash_block))

    def add_connection(self, from_block, from_idx, out_idx, to_block, to_idx, inp_idx):
        assert inp_idx < to_block.num_inputs()
        assert out_idx < from_block.num_outputs()
        hash_from, hash_from_block = self.get_hash(from_block, from_idx)
        hash_to, hash_to_block = self.get_hash(to_block, to_idx)
        assert hash_from != hash_to

        from_id = self.ids[hash_from]
        to_id = self.ids[hash_to]

        self.matrix[from_id][to_id] = np.array([inp_idx + 1])

class Block:
    def __init__(self, f: Callable, is_class: bool, max_queue: int, output_names: List[str]):
        self.f = f
        self.name = f.__name__
        self.is_class = is_class
        if self.is_class:
            input_args = dict(signature(self.f.run).parameters)
            del input_args['self']
        else:
            input_args = dict(signature(self.f).parameters)
        args = [(k, val.default) for k, val in input_args.items()]
        self.input_args = dict([(k, val) for k, val in args if val == _empty])
        self.custom_args = dict([(k, val) for k, val in args if val != _empty])
        self.max_queue = max_queue
        self.output_names = output_names if output_names != None else ['y']

    def num_inputs(self):
        return len(self.input_args)

    def num_outputs(self):
        return len(self.output_names)

    def __iter__(self):
        yield 'f', self.f
        yield 'name', self.name
        yield 'input_args', self.input_args
        yield 'max_queue', self.max_queue
        yield 'output_names', self.output_names

def block(f=None, max_queue=2, output_names=None, tag='None'):
    """
    Decorator function to tag custom blocks to be added to pipeline
    :param f: The function to tag (as a decorator it will be automatically passed)
    :param max_queue: Max queue length for the output of the block
    :param output_names: List of names for each output
    :param tag: Tag to organize decorated blocks
    :return:
    """
    if f is None:  # Correctly manage decorator duality
        return partial(block, max_queue=max_queue, output_names=output_names, tag=tag)

    if isinstance(f, types.FunctionType):
        if not isgeneratorfunction(f):
            raise TypeError('The function you tagged is not a generator')
        if signature(f).parameters and list(signature(f).parameters.keys())[0] == 'self':
            raise TypeError('The function you passed is a class method, we only support functions right now')
        is_class = False
    else:
        # Is a custom class so we need to process it differently (instantiate)
        assert hasattr(f, 'run'), 'The class %s you decorated must have a run method' % f.__name__
        if not isgeneratorfunction(f.run):
            raise TypeError('The function you tagged is not a generator')
        is_class = True

    pipeline.register_block(f, is_class, max_queue, output_names)
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
                q.put(ret[i])

class FakeQueue:
    def get(self):
        return []

class TerminableThread(Thread):
    # Thread class with a _stop() method.
    # The thread itself has to check
    # regularly for the stopped() condition.
    def __init__(self, f, thread_args=(), thread_kwargs={}, *args, **kwargs):
        super(TerminableThread, self).__init__(*thread_args, **thread_kwargs)
        self._stopper = Event()
        self.target = lambda: f(*args, **kwargs)

    def stop(self):
        self._stopper.set()

    def _stopped(self):
        return self._stopper.isSet()

    def run(self):
        while True:
            if self._stopped():
                return
            self.target()


pipeline = Pipeline()
