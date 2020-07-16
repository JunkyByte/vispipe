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


# Create pipeline
#p = Pipeline()
#
## Create nodes
#load_images = p.add_node('np_iter_file', path='tests/data/mnist.npy')
#load_labels = p.add_node('np_iter_file', path='tests/data/mnist_labels.npy')
#reshape = p.add_node('np_reshape', shape=(28, 28))
#resize = p.add_node('resize_cv2', width=56, height=56)
#batch_images = p.add_node('batchify/images', size=32)
#batch_labels = p.add_node('batchify/labels', size=32)
#
## Add connections
#p.add_conn(load_images, 0, reshape, 0)
#p.add_conn(reshape, 0, resize, 0)
#p.add_conn(resize, 0, batch_images, 0)
#p.add_conn(load_labels, 0, batch_labels, 0)
#
## Set outputs
#p.add_output(batch_images)
#p.add_output(batch_labels)
#
## Run it
#p.run(slow=False, use_mp=False)
#for batch_x, batch_y in zip(p.outputs['images'], p.outputs['labels']):
#    print(batch_x.shape, batch_y.shape)

# Save it
#p.save('./scratch_test.pickle')

#p.join()
#p.clear_pipeline()
#p.load('./test.pickle')
