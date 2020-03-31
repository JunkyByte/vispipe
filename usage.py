import vispipe
from vispipe import Pipeline
import numpy as np


@vispipe.block
def rand_range(minn=-1., maxx=1.):
    return (np.random.random() * (maxx - minn)) - minn


@vispipe.block(output_names=['y1', 'y2'])
def test_identity_2_out(input1):
    return input1, input1 + 1


@vispipe.block
def print_test(input1):
    msg = 'Value: %s' % input1
    print(msg)
    return msg


@vispipe.block(intercept_end=True, max_queue=1)
class accumulator:
    def __init__(self):
        self.i = 0

    def run(self, x):
        if x is StopIteration:
            self.i += 1
            if self.i < 5:
                return Pipeline._skip(self.i - 1)
            return self.i - 1
        return -1


#pipeline = Pipeline()

#img1 = pipeline.add_node('random_image')
#img2 = pipeline.add_node('random_image')
#add = pipeline.add_node('test_addition')
#plus1 = pipeline.add_node('test_plus1')
#plus2 = pipeline.add_node('test_plus100')
#timern = pipeline.add_node('benchmark', n=100000, log=True)

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
