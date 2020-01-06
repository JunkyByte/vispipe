from inspect import signature, isgeneratorfunction
from typing import List, Callable
from threading import Thread, Event
from queue import Queue

class Pipeline:
    def __init__(self):
        self._blocks = {}
        self.pipeline = []
        self.runner = PipelineRunner()

    def add_block(self, f : Callable, max_queue : int, output_names=None) -> None:
        block = Block(f, signature(f).parameters, max_queue, output_names)
        assert block.name not in self._blocks.keys()
        self._blocks[block.name] = block

    def add_pipeline(self, block):
        self.pipeline.append(block)

    def build(self) -> None:
        # ask runner to build the pipeline
        self.runner.build_pipeline(self.pipeline)

class PipelineRunner:
    def __init__(self):
        self.built = False
        self.threads = []
        self.in_queues = []
        self.out_queues = []

    def build_pipeline(self, pipeline):
        # Collect each block arguments, create connections between each other using separated threads
        # use queues to pass data between the threads.
        # The first block of the pipeline must be connected solely to an input generator
        self.built = True
        for block in pipeline:
            in_q = Queue(10)
            out_q = Queue(10)
            thr = TerminableThread(lambda: run_block(**dict(block), in_q=in_q, out_q=out_q))
            thr.daemon = True
            self.threads.append(thr)

        for thr in self.threads:
            thr.start()

        i = 0
        while True:
            import time
            in_q.put((i, i))
            i = list(out_q.get())[0]
            print(i)
            time.sleep(3)

class Block:
    def __init__(self, f: Callable, input_args: List[str], max_queue: int, output_names: List[str]):
        self.f = f
        self.name = f.__name__
        self.input_args = input_args
        self.max_queue = max_queue
        self.output_names = output_names if output_names != None else ['y']

    def __iter__(self):
        yield 'f', self.f
        yield 'name', self.name
        yield 'input_args', self.input_args
        yield 'max_queue', self.max_queue
        yield 'output_names', self.output_names

def block(f : callable, max_queue=2, output_names=None, tag='None'):
    """
    Decorator function to tag custom blocks to be added to pipeline
    :param f: The function to tag (as a decorator it will be automatically passed)
    :param max_queue: Max queue length for the output of the block
    :param output_names: List of names for each output
    :param tag: Tag to organize decorated blocks
    :return:
    """
    if not isgeneratorfunction(f):
        raise TypeError('The function you tagged is not a generator')
    if signature(f).parameters and list(signature(f).parameters.keys())[0] == 'self':
        raise TypeError('The function you passed is a class method, we only support functions right now')

    pipeline.add_block(f, max_queue, output_names)
    return f

def run_block(f, in_q, out_q, *args, **kwargs):
    x = in_q.get()
    ret = f(*x)
    out_q.put(ret)

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
