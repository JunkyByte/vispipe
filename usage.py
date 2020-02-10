from vispipe import vispipe
import numpy as np
import math

@vispipe.block
def custom_block_no_input():
    yield 1

@vispipe.block
def custom_block_rand():
    yield np.random.randn()

@vispipe.block
def custom_block_rand_range(min=-1., max=1.):
    yield (np.random.random() * (max - min)) - min

@vispipe.block
def custom_block_test_identity(input1):
    yield input1

@vispipe.block(output_names=['y1', 'y2'])
def custom_block_test_identity_2_out(input1):
    yield input1, input1 + 1

@vispipe.block
def custom_block_test_plus1(input1):
    yield input1 + 1

@vispipe.block
def custom_block_test_addition(input1, input2):
    yield input1 + input2

@vispipe.block
def custom_block_sin(input1):
    yield math.sin(input1)

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
def custom_block_print_test(input1):
    print('I am a print block and my value is %s' % input1)
    yield None

@vispipe.block
class custom_block_class:
    def __init__(self):
        self.last_element = None

    def run(self, input1):
        print('I am a print block CLASS and last value was %s while now is %s' % (self.last_element, input1))
        self.last_element = input1
        yield None


# Pipeline Test
custom_add = vispipe.pipeline._blocks['custom_block_test_addition']
custom_sin = vispipe.pipeline._blocks['custom_block_sin']
custom_noinp = vispipe.pipeline._blocks['custom_block_no_input']
custom_rand = vispipe.pipeline._blocks['custom_block_rand_range']
two_out = vispipe.pipeline._blocks['custom_block_test_identity_2_out']
out_test = vispipe.pipeline._blocks['custom_block_print_test']
out_test_class = vispipe.pipeline._blocks['custom_block_class']

#inp1 = vispipe.pipeline.add_node(custom_noinp)
rand = vispipe.pipeline.add_node(custom_rand, min=0, max=np.pi / 2)
rand2 = vispipe.pipeline.add_node(custom_rand, min=0, max=np.pi / 2)
#sumbetw = vispipe.pipeline.add_node(custom_add)
#output_print = vispipe.pipeline.add_node(out_test)
output_print = vispipe.pipeline.add_node(out_test_class)
output_print2 = vispipe.pipeline.add_node(out_test_class)
sin = vispipe.pipeline.add_node(custom_sin)
sin2 = vispipe.pipeline.add_node(custom_sin)

# Constant input are attached to sum which returns 2
#vispipe.pipeline.add_conn(custom_noinp, inp1, 0, custom_add, sumbetw, 0)
#vispipe.pipeline.add_conn(custom_rand, rand, 0, custom_add, sumbetw, 1)
#vispipe.pipeline.add_conn(custom_add, sumbetw, 0, custom_sin, sin, 0)
vispipe.pipeline.add_conn(custom_rand, rand, 0, custom_sin, sin, 0)
vispipe.pipeline.add_conn(custom_rand, rand2, 0, custom_sin, sin2, 0)
#vispipe.pipeline.add_conn(custom_sin, sin, 0, out_test, output_print, 0)
vispipe.pipeline.add_conn(custom_sin, sin, 0, out_test_class, output_print, 0)
vispipe.pipeline.add_conn(custom_sin, sin2, 0, out_test_class, output_print2, 0)

vispipe.pipeline.build()

#vispipe.pipeline.run()

print(0)
