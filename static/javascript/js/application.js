let app = new PIXI.Application({
    antialias: true,
    autoResize: true,
    resolution: window.devicePixelRatio
});
app.renderer.backgroundColor = 0x202125;
app.renderer.view.style.position = 'absolute';
app.renderer.view.style.display = 'block';
document.body.appendChild(app.view);
var WIDTH = app.renderer.width / app.renderer.resolution;
var HEIGHT = app.renderer.height / app.renderer.resolution;
var FONT = 'Arial';
var FONT_SIZE = 22;
var TEXT_COLOR = 'white';
var BUTTON_COLOR = 0xcfef92;
var BLOCK_COLOR = 0x5DBCD2;
//
var rect, text;
[rect, text] = draw_block('aaaa', 0, 0);
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
    WIDTH = app.renderer.width / app.renderer.resolution;
    HEIGHT = app.renderer.height / app.renderer.resolution;

}
resize();

// Pipeline class
var pipeline = new Pipeline();
var sidemenu = new SideMenu();
window.addEventListener('resize', function() { sidemenu.resize_menu()}, false);
window.addEventListener('mousewheel', function(ev){
    for (var i = 0; i < sidemenu.tags.length; i++){
        if (sidemenu.pane[sidemenu.tags[i]].visible === true){
            var new_y = sidemenu.pane[sidemenu.tags[i]].y += ev.wheelDelta / 5;
            new_y = Math.max(new_y, -(50 - HEIGHT + 50 * sidemenu.pane[sidemenu.tags[i]].children.length));
            sidemenu.pane[sidemenu.tags[i]].y = Math.min(0, new_y);
        }
    }
}, false)

$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('block', function(msg) {
        var block = new Block(msg.name, msg.input_args, msg.custom_args, msg.output_names, msg.tag);
        pipeline.STATIC_NODES.push(new StaticNode(block));
    });

    socket.on('end_block', function(msg) {
        sidemenu.populate_menu(pipeline, app);
    });

    //socket.emit('test_receive', 'test_send_see_me_python')
});
//$('#log').html(numbers_string);
