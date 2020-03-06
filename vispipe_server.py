from vispipe import vispipe
from flask_socketio import SocketIO
from threading import Thread, Event
from flask import Flask, render_template, session, jsonify
import usage
import numpy as np
import cv2
import traceback
import os

app = Flask(__name__)
SESSION_TYPE = 'redis'
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=False, engineio_logger=False)
PATH_CKPT = './scratch_test.pickle'


def share_blocks():
    for block in vispipe.pipeline.get_blocks(serializable=True):
        socketio.emit('new_block', block)
    socketio.emit('end_block', None)


@socketio.on('new_node')
def new_node(block):
    try:
        print('New node')
        block = vispipe.pipeline._blocks[block['name']]
        node_hash = hash(vispipe.pipeline.add_node(block))
        return {**{'id': node_hash}, **block.serialize()}, 200
    except Exception as e:
        print(traceback.format_exc())
        return str(e), 500


@socketio.on('remove_node')
def remove_node(id):
    try:
        print('Removing node')
        vispipe.pipeline.remove_node(id)
        return {}, 200
    except Exception as e:
        print(traceback.format_exc())
        return str(e), 500


@socketio.on('new_conn')
def new_conn(x):
    try:
        vispipe.pipeline.add_conn(x['from_hash'], x['out_idx'], x['to_hash'], x['inp_idx'])
        return {}, 200
    except Exception as e:
        print(traceback.format_exc())
        return str(e), 500


@socketio.on('set_custom_arg')
def set_custom_arg(data):
    try:
        vispipe.pipeline.set_custom_arg(data['id'], data['key'], data['value'])
        return {}, 200
    except Exception as e:
        print(traceback.format_exc())
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
            for node_hash, value in vis.items():
                node = vispipe.pipeline.get_node(int(node_hash))
                if node.block.data_type == 'image':
                    value, shape = process_image(value)
                elif node.block.data_type == 'raw':
                    if isinstance(value, (np.ndarray, list)):
                        value = np.around(value, 2)
                    elif isinstance(value, float):
                        value = round(value, 2)
                    value = str(value)

                socketio.emit('send_vis', {**{'id': node_hash, 'value': value}, **node.block.serialize()})
        except Exception as e:
            print(traceback.format_exc())
            socketio.emit('message', str(e))
        socketio.sleep(0.1)


@socketio.on('run_pipeline')
def run_pipeline():
    try:
        if not vispipe.pipeline.runner.built:
            vispipe.pipeline.build()

        vispipe.pipeline.run(slow=True)  # TODO: This becomes a parameter passed to the server (once wrapped)
        global thread
        thread_stop_event.clear()
        if not thread.isAlive():
            thread = socketio.start_background_task(send_vis)
        return {}, 200
    except Exception as e:
        print(traceback.format_exc())
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


@socketio.on('clear_pipeline')
def clear_pipeline():
    try:
        stop_pipeline()
        vispipe.pipeline.clear_pipeline()
        return {}, 200
    except Exception as e:
        print(traceback.format_exc())
        return str(e), 500


@socketio.on('save_nodes')
def save_nodes(vis_data):
    try:
        vispipe.pipeline.save(PATH_CKPT, vis_data)
        print('Saved checkpoint')
        return {}, 200
    except Exception as e:
        print(traceback.format_exc())
        return str(e), 500


def load_checkpoint(path):
    if not os.path.isfile(path):
        return

    _, vis_data = vispipe.pipeline.load(PATH_CKPT)
    pipeline = {'nodes': [], 'blocks': [], 'connections': [], 'custom_args': []}
    for node in vispipe.pipeline.nodes():
        pipeline['nodes'].append(hash(node))
        conn = vispipe.pipeline.connections(hash(node), out=True)
        pipeline['connections'].append([(hash(n), i, j) for n, i, j, _ in conn])
        pipeline['blocks'].append(node.block.serialize())
        pipeline['custom_args'].append(node.custom_args)
    socketio.emit('load_checkpoint', {'vis_data': vis_data, 'pipeline': pipeline})


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

    print('Loading checkpoint from %s' % PATH_CKPT)
    load_checkpoint(PATH_CKPT)

    socketio.emit('auto_save', None)

@app.route('/get_menu')
def get_menu():
    blocks = vispipe.pipeline.get_blocks()
    menu_data = {}
    for block in blocks:
        if block["tag"] in menu_data.keys():
            menu_data[block["tag"]]["items"].append({"name" : block["name"], "label" : block["name"]})
        else:
            menu_data[block["tag"]] = {"name" : block["tag"], "label" : block["tag"], "items" : []} 
    print(menu_data.values())
    return jsonify(menu_data)


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app)
