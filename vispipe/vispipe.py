from functools import partial
from inspect import signature, isgeneratorfunction, _empty
from typing import List, Callable
from threading import Thread, Event
from queue import Queue
import os
import types
import pickle
import numpy as np
import time
MAXSIZE = 100
assert np.log10(MAXSIZE) == int(np.log10(MAXSIZE))


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
        self.pipeline = PipelineGraph()
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

    def add_node(self, block, **kwargs):
        return self.pipeline.add_node(block, **kwargs)

    def remove_node(self, block, index):
        self.pipeline.remove_node(block, index)

    def clear_pipeline(self):
        self.pipeline.clear_pipeline()
        self.runner.unbuild()

    def add_conn(self, from_block, from_idx, out_idx, to_block, to_idx, inp_idx):
        self.pipeline.add_connection(
            from_block, from_idx, out_idx, to_block, to_idx, inp_idx)

    def set_custom_arg(self, block, index, key, value):
        self.pipeline.set_custom_arg(block, index, key, value)

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

    def read_vis(self):
        vis = {}
        idx = self.__vis_index()

        for key, consumer in self.vis_source.items():
            vis[key] = consumer.read(idx)

        return vis

    def __vis_index(self):
        return min([vis.size() for vis in self.vis_source.values()]) - 1

    def build_pipeline(self, pipeline):
        assert not self.built
        if len(pipeline.ids.keys()) == 0:
            return
        # Collect each block arguments, create connections between each other using separated threads
        # use queues to pass data between the threads.
        used_ids = np.array(list(pipeline.ids.values()))
        used_ids.sort()
        nodes = pipeline.nodes[used_ids]
        custom_args = pipeline.custom_args[used_ids]
        nodes_conn = np.array([x[used_ids] for x in pipeline.matrix[used_ids]])
        gain = np.array([n.num_outputs() / (n.num_inputs() + 1e-16)
                         for n in nodes])
        to_process = list(np.arange(len(used_ids))[(-gain).argsort()])
        trash = []

        i = 0
        while to_process != []:
            idx = to_process[i]

            node = nodes[idx]
            is_vis = True if node.tag == Pipeline.vis_tag else False

            custom_arg = custom_args[idx]
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
                instance_idx = pipeline.id_to_index(node, used_ids[idx])
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


class PipelineGraph:
    def __init__(self):
        self.matrix = np.empty(
            (MAXSIZE, MAXSIZE), dtype=object)  # Adjacency matrix
        self.nodes = np.empty((MAXSIZE,), dtype=Block)
        self.custom_args = np.empty((MAXSIZE,), dtype=dict)

        self.ids = {}  # Map from hash to assigned ids
        self.instances = {}  # Free ids for instances of same block
        self.free_idx = set([i for i in range(MAXSIZE)])  # Free ids for blocks

    def id_to_index(self, block, id):
        for k, v in self.ids.items():
            if v == id:
                hash_block = self.get_hash(block, index=0)[1]
                return k - hash_block * MAXSIZE
        assert False, 'The index has not been found'

    def get_hash(self, block, index=None):
        hash_block = hash(block)
        if hash_block not in self.instances.keys() and index is None:
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
        for i in range(MAXSIZE):
            self.matrix[id, i] = out
        return index

    def remove_node(self, block, index):
        hash_index, hash_block = self.get_hash(block, index=index)
        id = self.ids.pop(hash_index)
        self.nodes[id] = None  # unassign the node instance
        self.custom_args[id] = None
        self.matrix[id, :] = None  # delete the node connections
        self.free_idx.add(id)
        self.instances[hash_block].add(self.hash_to_instance(hash_index, hash_block))

    def clear_pipeline(self):
        self.matrix[:, :] = None
        self.nodes[:] = None
        self.custom_args[:] = None
        self.free_idx = set([i for i in range(MAXSIZE)])
        self.ids = {}

    def add_connection(self, from_block, from_idx, out_idx, to_block, to_idx, inp_idx):
        assert inp_idx < to_block.num_inputs()
        assert out_idx < from_block.num_outputs()
        assert from_block.tag != Pipeline.vis_tag  # Prevent connections from vis blocks
        hash_from, hash_from_block = self.get_hash(from_block, from_idx)
        hash_to, hash_to_block = self.get_hash(to_block, to_idx)
        assert hash_from != hash_to

        from_id = self.ids[hash_from]
        to_id = self.ids[hash_to]

        self.matrix[from_id][to_id] = np.array([inp_idx + 1])

    def set_custom_arg(self, block, index, key, value):
        hash_index, _ = self.get_hash(block, index)
        id = self.ids[hash_index]
        arg_type = self.nodes[id].custom_args_type[key]
        self.custom_args[id][key] = arg_type(value)


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
        self.skip = False

        if node.is_class:
            self.f = node.f().run
        else:
            self.f = node.f

    def run(self):
        # Pipeline.empty -> The function is not ready to return anything and you should skip its output
        # Pipeline._skip -> The function is still busy processing old input, do not pass another
        if self.skip:
            x = [Pipeline._empty for _ in range(len(self.in_q))]
            self.skip = False
        else:
            x = [q.get() for q in self.in_q]

        ret = list(self.f(*x, **self.custom_arg))
        if len(ret) == 1 and ret[0] == Pipeline._skip:
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
