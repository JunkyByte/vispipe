from vispipe import Pipeline
import vispipe.ops.flows
import vispipe.ops.inputs
import vispipe.ops.vis
import vispipe.ops.images
import vispipe.ops.common
import usage

from flask_socketio import SocketIO
from threading import Thread, Event
from flask import Flask, render_template, session
import numpy as np
import cv2
import traceback
import os
import logging

log = logging.getLogger('vispipe')
app = Flask(__name__)
SESSION_TYPE = 'redis'
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=False, engineio_logger=False)
PATH_CKPT = './scratch_test.pickle'
pipeline = Pipeline()


def share_blocks():
    for block in pipeline.get_blocks(serializable=True):
        socketio.emit('new_block', block)
    socketio.emit('end_block', None)


@socketio.on('new_node')
def new_node(block):
    try:
        log.info('New node')
        node_hash = hash(pipeline.add_node(block['name']))
        return {'id': node_hash}, 200
    except Exception as e:
        log.error(traceback.format_exc())
        return str(e), 500


@socketio.on('remove_node')
def remove_node(id):
    try:
        log.info('Removing node')
        pipeline.remove_node(id)
        return {}, 200
    except Exception as e:
        log.error(traceback.format_exc())
        return str(e), 500


@socketio.on('new_conn')
def new_conn(x):
    try:
        pipeline.add_conn(x['from_hash'], x['out_idx'], x['to_hash'], x['inp_idx'])
        return {}, 200
    except Exception as e:
        log.error(traceback.format_exc())
        return str(e), 500


@socketio.on('set_custom_arg')
def set_custom_arg(data):
    try:
        pipeline.set_custom_arg(data['id'], data['key'], data['value'])
        return {}, 200
    except Exception as e:
        log.error(traceback.format_exc())
        return str(e), 500


thread = Thread()
thread.daemon = True
thread_stop_event = Event()


def process_image(x):
    x = np.array(x, dtype=np.uint8)  # Cast to int
    if x.ndim in [0, 1, 4]:
        raise Exception('The format image you passed is not visualizable')

    if x.ndim == 2: # Convert from gray to rgb
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    if x.shape[-1] == 3:  # Add alpha channel
        x = np.concatenate([x, 255 * np.ones((x.shape[0], x.shape[1], 1))], axis=-1)
    shape = x.shape
    return np.reshape(x, (-1,)).tolist(), shape


def send_vis():
    while not thread_stop_event.isSet():
        try:
            vis = pipeline.runner.read_vis()
            for node_hash, value in vis.items():
                node = pipeline.get_node(int(node_hash))
                if node.block.data_type == 'image':
                    value, shape = process_image(value)
                elif node.block.data_type == 'raw':
                    if isinstance(value, (np.ndarray, list)):
                        try:
                            value = np.around(value, 2)
                        except Exception:
                            pass
                    elif isinstance(value, float):
                        value = round(value, 2)
                    value = str(value)

                socketio.emit('send_vis', {**{'id': node_hash, 'value': value}, **node.block.serialize()})
        except Exception as e:
            log.error(traceback.format_exc())
            socketio.emit('message', str(e))
        socketio.sleep(0.1)


@socketio.on('run_pipeline')
def run_pipeline():
    try:
        if not pipeline.runner.built:
            pipeline.build()

        pipeline.run(slow=True)  # TODO: This becomes a parameter passed to the server (once wrapped)
        global thread
        thread_stop_event.clear()
        if not thread.isAlive():
            thread = socketio.start_background_task(send_vis)
        return {}, 200
    except Exception as e:
        log.error(traceback.format_exc())
        return str(e), 500
    return 'Invalid State Encountered', 500


@socketio.on('stop_pipeline')
def stop_pipeline():
    try:
        global thread
        thread_stop_event.set()
        pipeline.unbuild()
        if thread.isAlive():
            thread.join()
        return {}, 200
    except Exception as e:
        return str(e), 500


@socketio.on('clear_pipeline')
def clear_pipeline():
    try:
        stop_pipeline()
        pipeline.clear_pipeline()
        return {}, 200
    except Exception as e:
        log.error(traceback.format_exc())
        return str(e), 500


@socketio.on('save_nodes')
def save_nodes(vis_data):
    try:
        pipeline.save(PATH_CKPT, vis_data)
        log.info('Saved checkpoint')
        return {}, 200
    except Exception as e:
        log.error(traceback.format_exc())
        return str(e), 500


def load_checkpoint(path):
    if not os.path.isfile(path):
        return

    _, vis_data = pipeline.load(PATH_CKPT)
    pipeline_def = {'nodes': [], 'blocks': [], 'connections': [], 'custom_args': []}
    for node in pipeline.nodes():
        pipeline_def['nodes'].append(hash(node))
        conn = pipeline.connections(hash(node), out=True)
        pipeline_def['connections'].append([(hash(n), i, j) for n, i, j, _ in conn])
        pipeline_def['blocks'].append(node.block.serialize())
        pipeline_def['custom_args'].append(node.custom_args)
    socketio.emit('load_checkpoint', {'vis_data': vis_data, 'pipeline': pipeline_def})


@app.route('/')
def index():
    session['test_session'] = 42
    return render_template('index.html')


@app.route('/get/')
def show_session():
    log.info(session['test_session'])
    return '%s' % session.get('test_session')


@socketio.on('connect')
def test_connect():
    # need visibility of the global thread object
    log.warning('Client connected')
    pipeline.clear_pipeline()

    log.info('Sharing blocks')
    share_blocks()

    log.info('Loading checkpoint from %s' % PATH_CKPT)
    load_checkpoint(PATH_CKPT)

    socketio.emit('set_auto_save', True)


@socketio.on('disconnect')
def test_disconnect():
    log.warning('Client disconnected')


if __name__ == '__main__':
    socketio.run(app)
