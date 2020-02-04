let app = new PIXI.Application({
    antialias: true,
    autoResize: true,
    resolution: window.devicePixelRatio
});
app.renderer.backgroundColor = 0xc4c2be;
app.renderer.view.style.position = 'absolute';
app.renderer.view.style.display = 'block';
document.body.appendChild(app.view);

//
var rect = draw_block('test0', 0, 0);
rect.position.set(100, 100);

rect.interactive = true;
rect.buttonMode = true;
rect
    // events for drag start
    .on('mousedown', onDragStart)
    .on('touchstart', onDragStart)
    // events for drag end
    .on('mouseup', onDragEnd)
    .on('mouseupoutside', onDragEnd)
    .on('touchend', onDragEnd)
    .on('touchendoutside', onDragEnd)
    // events for drag move
    .on('mousemove', onDragMove)
    .on('touchmove', onDragMove);

app.stage.addChild(rect);
//

// Listen for window resize events
window.addEventListener('resize', resize);

// Resize function window
function resize() {
    app.renderer.resize(window.innerWidth, window.innerHeight);
}
resize();

// TODO: Please change me in favor of a pipeline class that manages both static and actual nodes
var STATIC_NODES = [];

$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('block', function(msg) {
        var block = new Block(msg.name, msg.input_args, msg.custom_args, msg.output_names);
        STATIC_NODES.push(new StaticNode(block));
    });

    //socket.emit('test_receive', 'test_send_see_me_python')
});
//$('#log').html(numbers_string);
