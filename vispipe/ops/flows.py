from vispipe.vispipe import Pipeline
from vispipe.vispipe import block
"""
Flow generators allow to modify the way a stream of your pipeline is processed.
"""


@block(tag='flows')
class iterator:
    """
    Yields each element of its input, will take care of not accepting new inputs
    until it reaches a StopIteration.
    It will return StopIteration after the last element of its input.
    """

    def __init__(self):
        self.iterator = None
        self.next = StopIteration

    def run(self, x):
        y = self.next
        if y == StopIteration:
            self.iterator = iter(x)
            y = next(self.iterator)

        try:
            self.next = next(self.iterator)
        except StopIteration:
            self.next = StopIteration

        if self.next is not StopIteration:
            y = Pipeline._skip(y)

        yield y
