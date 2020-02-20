from vispipe import vispipe
from flask_socketio import SocketIO
from threading import Thread, Event
from flask import Flask, render_template
from flask import session
import usage
import numpy as np

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
        block = vispipe.pipeline._blocks[block['name']]
        id = vispipe.pipeline.add_node(block)
        return {**{'id': id}, **block.serialize()}
    except Exception as e:
        return str(e)


@socketio.on('new_conn')
def new_conn(x):
    try:
        from_block = vispipe.pipeline._blocks[x['from_block']['name']]
        to_block = vispipe.pipeline._blocks[x['to_block']['name']]
        vispipe.pipeline.add_conn(from_block, x['from_idx'], x['out_idx'], to_block, x['to_idx'], x['inp_idx'])
        return 200
    except Exception as e:
        return str(e)


thread = Thread()
thread_stop_event = Event()


def send_vis():
    while not thread_stop_event.isSet():
        for key, consumer in vispipe.pipeline.runner.vis_source.items():  # TODO: This access can be improved
            value = consumer.read()
            block, id = key
            if block.data_type == 'image':
                value = np.array(value, dtype=np.int)
                shape = value.shape
                value = np.reshape(value, (-1,)).tolist()  # TODO: Automatically manage non alpha images (concatenate internally) + find a way for grayscale images
            # TODO: MANAGE OTHER TYPES OF DATA (RAW)
            if not len(shape) == 3:
                socketio.emit('message', 'The value is not an image and will not be visualized')
            else:
                socketio.emit('send_vis', {**{'id': id, 'value': value}, **block.serialize()})
        socketio.sleep(1)


@socketio.on('run_pipeline')
def run_pipeline():
    if not vispipe.pipeline.runner.built:
        vispipe.pipeline.build()

    try:
        vispipe.pipeline.run()
        global thread
        assert not thread.isAlive()
        if len(vispipe.pipeline.runner.vis_source):
            thread_stop_event.clear()
            thread = socketio.start_background_task(send_vis)
        return 200
    except Exception as e:
        return str(e)

    assert False  # TODO get here sometimes


@socketio.on('stop_pipeline')
def stop_pipeline():
    try:
        vispipe.pipeline.stop()
        vispipe.pipeline.unbuild()
        thread_stop_event.set()
        return 200
    except Exception as e:
        return str(e)


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
