from vispipe import vispipe
import usage
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context
from random import random
from threading import Thread, Event
from flask import session
import itertools

app = Flask(__name__)
SESSION_TYPE = 'redis'
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

#random number Generator Thread
#thread = Thread()
#thread_stop_event = Event()

#def socket_blocks():
#    while not thread_stop_event.isSet():
#        socketio.emit('block_info', {'number': number}, namespace='/test')
#        socketio.sleep(5)

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

    #if not thread.isAlive():
        #print("Starting Thread")
        #thread = socketio.start_background_task(socket_blocks)


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app)
