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
var VIS_IMAGE_SIZE = 128;
var FONT = 'Arial';
var FONT_SIZE = 18;
var TEXT_COLOR = 'white';
var BUTTON_COLOR = 0x5DBCD2;
var BLOCK_COLOR = 0x5DBCD2;
var INPUT_COLOR = 0x5DBCD2;  // TODO: Add me
var OUTPUT_COLOR = 0x5DBCD2;

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
var runmenu = new RunMenu();
window.addEventListener('resize', function() {runmenu.resize_menu()}, false);
window.addEventListener('resize', function() {sidemenu.resize_menu()}, false);
window.addEventListener('mousewheel', function(ev){sidemenu.scroll_blocks(ev)}, false);

var socket;
$(document).ready(function(){
    socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('new_block', function(msg) {
        var block = new Block(msg.name, msg.input_args, msg.custom_args, msg.output_names, msg.tag, msg.data_type);
        pipeline.STATIC_NODES.push(new StaticNode(block));
    });

    socket.on('end_block', function(msg) {
        sidemenu.populate_menu(pipeline, app);
    });

    socket.on('send_vis', function(msg) {
        var id = msg.id;
        var name = msg.name;

        value = new Uint8Array(msg.value);
        var size = value.length / 4;
        var s = Math.sqrt(size);

        let texture = PIXI.Texture.fromBuffer(value, s, s);
        pipeline.vis[id + '-' + name].update_texture(texture);
    });

    socket.on('message', function(msg) {
        console.log(msg);
    });

    //socket.emit('test_receive', 'test_send_see_me_python')
});
//$('#log').html(numbers_string);

