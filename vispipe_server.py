from vispipe import vispipe
import usage
from flask_socketio import SocketIO
from flask import Flask, render_template
from flask import session

app = Flask(__name__)
SESSION_TYPE = 'redis'
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)


def share_blocks():
    for block in vispipe.pipeline.get_blocks(serializable=True):
        socketio.emit('new_block', block)
    socketio.emit('end_block', None)


@socketio.on('new_node')
def new_node(block):
    block = vispipe.pipeline._blocks[block['name']]
    id = vispipe.pipeline.add_node(block)
    socketio.emit('node_id', {**{'id': id}, **block.serialize()})


@socketio.on('new_conn')
def new_conn(x):
    from_block = vispipe.pipeline._blocks[x['from_block']['name']]
    to_block = vispipe.pipeline._blocks[x['to_block']['name']]
    vispipe.pipeline.add_conn(from_block, x['from_idx'], x['out_idx'], to_block, x['to_idx'], x['inp_idx'])


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
    print('Sharing blocks')
    share_blocks()

    #
    import numpy as np
    arr = (np.ones((100 * 100 * 4), dtype=np.int) * 255).tolist()
    socketio.emit('test_send', {'x': arr})


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app)
