import threading
from threading import Thread
from queue import Queue
import time

class TerminableThread(Thread):

    # Thread class with a _stop() method.
    # The thread itself has to check
    # regularly for the stopped() condition.

    def __init__(self, target, thread_args=(), thread_kwargs={}, *args, **kwargs):
        super(TerminableThread, self).__init__(*thread_args, **thread_kwargs)
        self._stopper = threading.Event()
        self.target = lambda: target(*args, **kwargs)

    def stop(self):
       self._stopper.set()

    def _stopped(self):
        return self._stopper.isSet()

    def run(self):
        while True:
            if self._stopped():
                return
            self.target()

t1 = TerminableThread(lambda x, *args, **kwargs: print(x), x=1, y=3, z=2)
t1.start()
time.sleep(1)
t1.stop()
t1.join()
