from vispipe import vispipe
from flask_socketio import SocketIO
from threading import Thread, Event
from flask import Flask, render_template, session
import usage
import numpy as np
import cv2

app = Flask(__name__)
SESSION_TYPE = 'redis'
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=False, engineio_logger=False)


def share_blocks():
    for block in vispipe.pipeline.get_blocks(serializable=True):
        socketio.emit('new_block', block)
    socketio.emit('end_block', None)


@socketio.on('new_node')
def new_node(block):
    try:
        print('New node')
        block = vispipe.pipeline._blocks[block['name']]
        id = vispipe.pipeline.add_node(block)
        return {**{'id': id}, **block.serialize()}, 200
    except Exception as e:
        return str(e), 500


@socketio.on('remove_node')
def remove_node(data):
    try:
        print('Removing node')
        block_dict = data['block']
        index = data['index']
        block = vispipe.pipeline._blocks[block_dict['name']]
        vispipe.pipeline.remove_node(block, index)
        return {}, 200
    except Exception as e:
        return str(e), 500


@socketio.on('new_conn')
def new_conn(x):
    try:
        from_block = vispipe.pipeline._blocks[x['from_block']['name']]
        to_block = vispipe.pipeline._blocks[x['to_block']['name']]
        vispipe.pipeline.add_conn(from_block, x['from_idx'], x['out_idx'], to_block, x['to_idx'], x['inp_idx'])
        return {}, 200
    except Exception as e:
        return str(e), 500


@socketio.on('set_custom_arg')
def set_custom_arg(data):
    try:
        block_dict = data['block']
        block = vispipe.pipeline._blocks[block_dict['name']]
        vispipe.pipeline.set_custom_arg(block, data['id'], data['key'], data['value'])
        return {}, 200
    except Exception as e:
        return str(e), 500


thread = Thread()
thread.daemon = True
thread_stop_event = Event()


def process_image(x):
    x = np.array(x, dtype=np.int)  # Cast to int
    if x.ndim in [0, 1, 4]:
        raise Exception('The format image you passed is not visualizable')

    if x.ndim == 2:  # Add channel dim
        x = x.reshape((*x.shape, 1))
    if x.shape[-1] == 1:  # Convert to rgb a grayscale
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    if x.shape[-1] == 3:  # Add alpha channel
        x = np.concatenate([x, np.ones((x.shape[0], x.shape[1], 1))], axis=-1)
    shape = x.shape
    return np.reshape(x, (-1,)).tolist(), shape


def send_vis():
    while not thread_stop_event.isSet():
        try:
            vis = vispipe.pipeline.runner.read_vis()
            for key, value in vis.items():
                block, id = key
                if block.data_type == 'image':
                    value, shape = process_image(value)
                elif block.data_type == 'raw':
                    if isinstance(value, (np.array, list)):
                        value = np.around(value, 2)
                    elif isinstance(value, float):
                        value = round(value, 2)
                    value = str(value)

                socketio.emit('send_vis', {**{'id': id, 'value': value}, **block.serialize()})
        except Exception as e:
            socketio.emit('message', str(e))

        socketio.sleep(1)


@socketio.on('run_pipeline')
def run_pipeline():
    try:
        if not vispipe.pipeline.runner.built:
            vispipe.pipeline.build()

        vispipe.pipeline.run()
        global thread
        assert not thread.isAlive()
        if len(vispipe.pipeline.runner.vis_source):
            thread_stop_event.clear()
            thread = socketio.start_background_task(send_vis)
        return {}, 200
    except Exception as e:
        return str(e), 500
    return 'Invalid State Encountered', 500


@socketio.on('stop_pipeline')
def stop_pipeline():
    try:
        global thread
        thread_stop_event.set()
        vispipe.pipeline.unbuild()
        if thread.isAlive():
            thread.join()
        return {}, 200
    except Exception as e:
        return str(e), 500


@app.route('/')
def index():
    session['test_session'] = 42
    return render_template('index.html')


@app.route('/get/')
def show_session():
    print(session['test_session'])
    return '%s' % session.get('test_session')


@socketio.on('connect')
def test_connect():
    # need visibility of the global thread object
    print('Client connected')
    vispipe.pipeline.clear_pipeline()

    print('Sharing blocks')
    share_blocks()

    #
    #import numpy as np
    #arr = (np.ones((100 * 100 * 4), dtype=np.int) * 255).tolist()
    #socketio.emit('test_send', {'x': arr})


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app)
