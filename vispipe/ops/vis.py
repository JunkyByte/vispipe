from vispipe.vispipe import block


@block(tag='vis', max_queue=10, data_type='image')
def test_vis(input1):
    yield input1


@block(tag='vis', max_queue=100, data_type='raw')
def test_vis_raw(input1):
    yield input1

