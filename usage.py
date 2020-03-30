import vispipe
from vispipe import Pipeline
import numpy as np
import math
import time
import logging


@vispipe.block
def three_args(arg1=5, arg2='ciao', arglonglong='7x7'):
    yield arg1


@vispipe.block
def some_list():
    yield [0, 1, 2, 3, 4, 5, 42]


@vispipe.block
def five_args(arg1=5, arg2='ciao', arglong='7x7', longveryarg=1023, onemore=-100.3):
    yield 1


@vispipe.block
def no_input():
    yield 1


@vispipe.block
def image():
    yield np.concatenate([np.random.randint(0, 255, size=(28, 28, 3)), np.ones((28, 28, 1)) * 255], axis=-1)


@vispipe.block
def image_highres():
    yield np.concatenate([np.random.randint(0, 255, size=(1024, 1024, 3)), np.ones((1024, 1024, 1)) * 255], axis=-1)


@vispipe.block
def image_randsize():
    s1 = np.random.randint(32, 256)
    s2 = np.random.randint(32, 256)
    yield np.concatenate([np.random.randint(0, 255, size=(s1, s2, 3)), np.ones((s1, s2, 1)) * 255], axis=-1)


@vispipe.block
def image_randw():
    s1 = 256
    s2 = np.random.randint(32, 256)
    yield np.concatenate([np.random.randint(0, 255, size=(s1, s2, 3)), np.ones((s1, s2, 1)) * 255], axis=-1)


@vispipe.block
def image_randh():
    s1 = np.random.randint(32, 256)
    s2 = 256
    yield np.concatenate([np.random.randint(0, 255, size=(s1, s2, 3)), np.ones((s1, s2, 1)) * 255], axis=-1)


@vispipe.block
def image_plus(x):
    yield np.concatenate([x * np.ones((28, 28, 3)), np.ones((28, 28, 1)) * 255], axis=-1)


@vispipe.block
def image_rgb(r, g, b):
    ones = np.ones((28, 28, 1))
    yield np.concatenate([r * ones, g * ones, b * ones, ones * 255], axis=-1)


@vispipe.block
def test_plus100(x):
    yield x + 100


@vispipe.block
def multiply100(x):
    yield x * 100


@vispipe.block
def rand():
    yield np.random.randn()


@vispipe.block
def nparray(arr: np.array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])):
    yield arr


@vispipe.block
def randint():
    yield np.random.randint(0, 255)


@vispipe.block
def rand_range(minn=-1., maxx=1.):
    yield (np.random.random() * (maxx - minn)) - minn


@vispipe.block
def test_identity(input1):
    yield input1


@vispipe.block(output_names=['y1', 'y2'])
def test_identity_2_out(input1):
    yield input1, input1 + 1


@vispipe.block
def test_plus1(input1):
    yield input1 + 1


@vispipe.block
def test_addition(input1, input2):
    yield input1 + input2


@vispipe.block
def custom_high_gain_test(input1):
    yield input1


@vispipe.block
def another_1(input1):
    yield 0


@vispipe.block
def another_2(input1):
    yield 0


@vispipe.block
def another_3(input1):
    yield 0


@vispipe.block
def another_4(input1):
    yield 0


@vispipe.block
def in_5(input1, input2, input3, input4, input5):
    yield 0


@vispipe.block
def in_2(input1, input2):
    yield 0


@vispipe.block
def in_3(input1, input2, input3):
    yield 0


@vispipe.block
def print_test(input1):
    msg = 'Value: %s' % input1
    print(msg)
    yield msg


@vispipe.block
class classex:
    def __init__(self):
        self.last_element = None

    def run(self, input1):
        print('I am a print block CLASS and last value was %s while now is %s' % (
            self.last_element, input1))
        self.last_element = input1
        yield None


@vispipe.block
class testempty:
    def __init__(self):
        self.count = 0
        self.sum = 0

    def run(self, input1):
        if self.count == 10:
            val = self.sum
            self.sum = 0
            self.count = 0
            yield val
        else:
            self.sum += input1
            self.count += 1
            yield vispipe.pipeline._empty


@vispipe.block
class timer_out:
    def __init__(self):
        self.count = 0

    def run(self, x):
        self.count += 1
        if self.count == 5:
            raise StopIteration
        yield x


@vispipe.block(intercept_end=True, max_queue=1)
class accumulator:
    def __init__(self):
        self.i = 0

    def run(self, x):
        if x is StopIteration:
            self.i += 1
            if self.i < 5:
                yield Pipeline._skip(self.i - 1)
            yield self.i - 1
        yield -1


pipeline = Pipeline()

for i in range(100):
    pipeline.add_node('image')

pipeline.clear_pipeline()

#img1 = pipeline.add_node('image')
#img2 = pipeline.add_node('image')
#add = pipeline.add_node('test_addition')
#plus1 = pipeline.add_node('test_plus1')
#plus2 = pipeline.add_node('test_plus100')
#timern = pipeline.add_node('benchmark', n=1000)

#pipeline.add_conn(img1, 0, add, 0)
#pipeline.add_conn(img2, 0, add, 1)
#pipeline.add_conn(add, 0, plus1, 0)
#pipeline.add_conn(plus1, 0, plus2, 0)
#pipeline.add_conn(plus2, 0, timern, 0)

#pipeline.add_macro(add, timern)

#pipeline.run(slow=False, use_mp=False)
#pipeline.join()

#pipeline.clear_pipeline()
#pipeline.save('./scratch_test.pickle')
#pipeline.load('./test.pickle')
