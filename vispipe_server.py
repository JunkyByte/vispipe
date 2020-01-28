from vispipe import vispipe
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context
from random import random
from threading import Thread, Event
from flask import session

app = Flask(__name__)
SESSION_TYPE = 'redis'
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

#random number Generator Thread
thread = Thread()
thread_stop_event = Event()

@socketio.on('test_receive', namespace='/test')
def show_msg(msg):
    print(msg)

def socket_blocks():
    while not thread_stop_event.isSet():
        number = round(random()*10, 3)
        print(number)
        socketio.emit('newnumber', {'number': number}, namespace='/test')
        socketio.sleep(5)

@app.route('/')
def index():
    session['test_session'] = 42
    return render_template('index.html')

@app.route('/get/')
def show_session():
    print(session['test_session'])
    return '%s' % session.get('test_session')

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

    #Start the random number generator thread only if the thread has not been started before.
    if not thread.isAlive():
        print("Starting Thread")
        thread = socketio.start_background_task(socket_blocks)

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, debug=True)