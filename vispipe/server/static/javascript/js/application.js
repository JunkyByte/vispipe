let app = new PIXI.Application({
    antialias: false,  // This makes a huge difference
    autoResize: true,
    resolution: window.devicePixelRatio
});
app.renderer.backgroundColor = 0x202125;
app.renderer.view.style.position = 'absolute';
app.renderer.view.style.display = 'block';
document.body.appendChild(app.view);

var viewport = new Viewport.Viewport({
    screenWidth: window.innerWidth,
    screenHeight: window.innerHeight,
    worldWidth: 3500,
    worldHeight: 2000,
    disableOnContextMenu: true,

    interaction: app.renderer.plugins.interaction
})
app.stage.addChild(viewport);
viewport
    .drag({wheel: false, mouseButtons: 'left', keyToPress: ['ControlLeft']})
    .pinch()
    .wheel()
    .clamp({right: true, bottom: true})
    .decelerate({friction: 0.85})
    .clampZoom({minWidth: 300, minHeight: 300, maxWidth: 2500, maxHeight: 2500})
viewport.plugins.get('wheel').pause()
window.addEventListener('wheel', function(ev) { if (ev.ctrlKey) { viewport.plugins.get('wheel').resume() }});
app.renderer.render(viewport)

var WIDTH = app.renderer.width / app.renderer.resolution;
var HEIGHT = app.renderer.height / app.renderer.resolution;
var VIS_IMAGE_SIZE = 128;
var VIS_RAW_SIZE = 256;
var CUSTOM_ARG_SIZE = 350;
var FONT = 'Arial';
var FONT_SIZE = 18;
var VIS_FONT_SIZE = 18;
var TEXT_COLOR = 'white';
var BUTTON_COLOR = 0x5DBCD2;
var BLOCK_COLOR = 0x5DBCD2;
var INPUT_COLOR = 0x5DBCD2;  // TODO: Add me
var INPUT_WRONG_COLOR = 0xED1909;
var INPUT_TEXT_COLOR = 0x26272E;
var OUTPUT_COLOR = 0x5DBCD2;

var autosave = false;

// Listen for window resize events
window.addEventListener('resize', resize);

// Resize function window
function resize() {
    viewport.screenWidth = window.innerWidth;
    viewport.screenHeight = window.innerHeight;
    app.renderer.resize(window.innerWidth, window.innerHeight);
    WIDTH = app.renderer.width / app.renderer.resolution;
    HEIGHT = app.renderer.height / app.renderer.resolution;
}
resize();

// Pipeline class
var pipeline = new Pipeline();
var sidemenu = new SideMenu();
var runmenu = new RunMenu();
var popupmenu = new PopupMenu();
window.addEventListener('resize', function() {runmenu.resize_menu()}, false);
window.addEventListener('resize', function() {sidemenu.resize_menu()}, false);
window.addEventListener('wheel', function(ev){sidemenu.scroll_blocks(ev)}, false);

var socket;
$(document).ready(function(){
    socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('new_block', function(msg) {
        var block = new Block(msg.name, msg.input_args, msg.custom_args, msg.custom_args_type,
                              msg.output_names, msg.tag, msg.data_type);
        pipeline.STATIC_NODES.push(new StaticNode(block));
    });

    socket.on('end_block', function(msg) {
        sidemenu.populate_menu(pipeline);
    });

    socket.on('send_vis', function(msg) {
        var data_type = msg.data_type;
        var vis_node = pipeline.vis[msg.id]

        if (data_type == 'image') {
            var value = new Uint8Array(msg.value);
            var size = value.length / 4;
            var s = Math.sqrt(size);
            var texture = PIXI.Texture.fromBuffer(value, s, s);
            vis_node.update_texture(texture);
        } else if (data_type == 'raw') {
            vis_node.update_text(msg.value);
        }
    });

    socket.on('message', function(msg) {
        console.log(msg);
    });

    socket.on('set_auto_save', function(msg){
        autosave = msg;
    });

    // Noted this is not in the event
    autosave = setInterval(() => {
        if (!autosave){
            return;
        }

        positions = [];
        for (var i=0; i<pipeline.DYNAMIC_NODES.length; i++){
            obj = pipeline.DYNAMIC_NODES[i];
            pos = obj.rect.position;
            positions.push([obj.id, pos.x, pos.y]);
        }
        socket.emit('save_nodes', positions, function(response, status){
            if (status !== 200){
                console.log(response);
            }
        });
    }, 30000);

    socket.on('load_checkpoint', function(msg){ // TODO: IMPORTANT fix multiple output not connected after reload
        var vis_data = msg.vis_data;            // TODO: FIX CUSTOM ARG SETTINGS FOR ITERATOR NOT WORKING
        var pipeline_def = msg.pipeline;
        var nodes = pipeline_def.nodes;
        var blocks = pipeline_def.blocks;
        var custom_args = pipeline_def.custom_args;
        var connections = pipeline_def.connections;

        // Create connections dict
        var conn_dict = {};
        var conn, hash;
        for (var i=0; i<connections.length; i++){
            conn = connections[i];
            if (conn.length == 0){
                continue;
            }
            hash = nodes[i];
            conn_dict[hash] = conn;
        }

        // Create positions dict
        var pos_dict = {};
        var pos;
        for (i=0; i<vis_data.length; i++){
            pos = vis_data[i];
            if (pos.length == 0){
                continue;
            }
            hash = pos[0];
            pos_dict[hash] = [pos[1], pos[2]];
        }

        var obj;
        var block, block_dict, arg, j, key;
        for (i=0; i<nodes.length; i++){
            // Create blocks
            block_dict = blocks[i];
            block = new Block(block_dict.name, block_dict.input_args, block_dict.custom_args,
                              block_dict.custom_args_type, block_dict.output_names,
                              block_dict.tag, block_dict.data_type);
            obj = pipeline.spawn_node_visual(block, nodes[i]);
            obj.rect.position.set(pos_dict[nodes[i]][0], pos_dict[nodes[i]][1]);

            for (j=0; j<Object.keys(custom_args[i]).length; j++){
                key = Object.keys(custom_args[i])[j];
                arg = Object.values(custom_args[i])[j];
                pipeline.set_custom_arg(obj, key, arg);
            }
        }

        // Create actual connections
        var conn, to, from, to_pos, from_pos;
        for (i=0; i<pipeline.DYNAMIC_NODES.length; i++){
            node = pipeline.DYNAMIC_NODES[i];

            conn = conn_dict[node.id];
            if (conn !== undefined) {
                for (j=0; j<conn.length; j++){
                    to_node = pipeline.find_node(conn[j][0]);
                    from = node.out_c[conn[j][1]]
                    to = to_node.in_c[conn[j][2]]

                    connection = create_connection(to, from); 
                    viewport.addChildAt(connection, viewport.children.length);

                    app.renderer.render(node.rect);  // Force rendering to update positions
                    app.renderer.render(to_node.rect);
                }
                update_all_lines(node);
            }

        }

        viewport.fitWorld(false)
        viewport.fitWidth(1500, true);
    });

    //socket.emit('test_receive', 'test_send_see_me_python')
});
//$('#log').html(numbers_string);

