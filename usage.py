from vispipe import vispipe

@vispipe.block
def custom_block_no_input():
    yield 1

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
def custom_high_gain_test(input1):
    yield input1

# Pipeline 1
# Has 2 input node / 2 add nodes, one removed node
custom_add = vispipe.pipeline._blocks['custom_block_test_addition']
custom_noinp = vispipe.pipeline._blocks['custom_block_no_input']
two_out = vispipe.pipeline._blocks['custom_block_test_identity_2_out']
high_g = vispipe.pipeline._blocks['custom_high_gain_test']
inp1 = vispipe.pipeline.add_node(custom_noinp)
inp2 = vispipe.pipeline.add_node(custom_noinp)
sum_full1 = vispipe.pipeline.add_node(custom_add)
sum_full2 = vispipe.pipeline.add_node(custom_add)
vispipe.pipeline.remove_node(custom_add, sum_full2)
sum_full2 = vispipe.pipeline.add_node(custom_add)
# Unused input
unusedinp = vispipe.pipeline.add_node(custom_noinp)
two_out_idx = vispipe.pipeline.add_node(two_out)
# Add one node with high gain that has a lower gain as dependency
high_gain = vispipe.pipeline.add_node(high_g)
# Add 1 is connected (both inputs) to the input
vispipe.pipeline.add_conn(custom_noinp, inp1, 0, custom_add, sum_full1, 0)
vispipe.pipeline.add_conn(custom_noinp, inp2, 0, custom_add, sum_full1, 1)
# Add 2 has one input connected to the input node and one connected to Add 1
vispipe.pipeline.add_conn(custom_noinp, inp1, 0, custom_add, sum_full2, 0)
vispipe.pipeline.add_conn(custom_add, sum_full1, 0, custom_add, sum_full2, 1)
# Two out has one input connected and it's output are not connected
vispipe.pipeline.add_conn(custom_noinp, inp1, 0, two_out, two_out_idx, 0)
# High gain, fully connected but has dependencies
vispipe.pipeline.add_conn(custom_add, sum_full2, 0, high_g, high_gain, 0)

# build the pipeline
vispipe.pipeline.build()

print(0)
