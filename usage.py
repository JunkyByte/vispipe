from vispipe import vispipe

@vispipe.block
def custom_block_test_identity(input1):
    yield input1

@vispipe.block(output_names=['y1', 'y2'])
def custom_block_test_identity_2_out(input1, input2):
    yield (input1, input2)

@vispipe.block
def custom_block_test_plus1(input1):
    yield input1 + 1

@vispipe.block
def custom_block_test_addition(input1, input2):
    yield input1 + input2

vispipe.pipeline.add_node(vispipe.pipeline._blocks['custom_block_test_addition'])
vispipe.pipeline.add_node(vispipe.pipeline._blocks['custom_block_test_addition'])
vispipe.pipeline.add_node(vispipe.pipeline._blocks['custom_block_test_addition'])
vispipe.pipeline.add_node(vispipe.pipeline._blocks['custom_block_test_plus1'])
vispipe.pipeline.add_node(vispipe.pipeline._blocks['custom_block_test_identity_2_out'])
#vispipe.pipeline.build()

print(0)
