from vispipe import block


@block(tag='vis', max_queue=1, data_type='image')
def vis_image(input1):
    yield input1


@block(tag='vis', max_queue=1, data_type='raw')
def vis_text(input1):
    yield input1


@block(tag='vis', max_queue=1, data_type='raw')
def vis_shape(input1):
    yield str(input1.shape)
