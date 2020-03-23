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
    .drag({wheel: false, mouseButtons: 'left', keyToPress: ['ControlLeft', 'ControlRight']})
    .pinch()
    .wheel()
    .clamp({right: true, bottom: true})
    .decelerate({friction: 0.85})
    .clampZoom({minWidth: 300, minHeight: 300, maxWidth: 2500, maxHeight: 2500})
viewport.plugins.get('wheel').pause()
window.addEventListener('wheel', function(ev) { if (ev.ctrlKey) {
    viewport.plugins.get('wheel').resume();
    clearTimeout(ev.wheeling);
    ev.wheeling = setTimeout(function() {
        ev.wheeling = undefined;
        viewport.plugins.get('wheel').pause();
    }, 200);
}});

app.renderer.render(viewport)
viewport.fitWorld(false)
viewport.fitWidth(1500, true);

var WIDTH = app.renderer.width / app.renderer.resolution;
var HEIGHT = app.renderer.height / app.renderer.resolution;
var VIS_IMAGE_SIZE = 256;
var VIS_RAW_SIZE = 256;
var CUSTOM_ARG_SIZE = 350;
var FONT = 'Arial';
var FONT_SIZE = 18;
var VIS_FONT_SIZE = 18;
var TEXT_COLOR = 'white';
var BUTTON_COLOR = 0x5DBCD2;
var BLOCK_COLOR = 0x5DBCD2;
var BLOCK_OUT_COLOR = 0xFF7256;
var INPUT_COLOR = 0x3fc32a;
var OUTPUT_COLOR = 0xc32a2a;
var INPUT_TEXT_COLOR = 0xd2757b;

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

    socket.on('end_block', function() {
        sidemenu.populate_menu(pipeline);
    });

    socket.on('send_vis', function(msg) {
        var data_type = msg.data_type;
        var vis_node = pipeline.vis[msg.id]

        if (data_type == 'image' || data_type == 'plot') {
            var shape = msg.shape
            var value = new Uint8Array(msg.value);
            var texture = PIXI.Texture.fromBuffer(value, shape[0], shape[1]);
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

        positions = {};
        for (var i=0; i<pipeline.DYNAMIC_NODES.length; i++){
            obj = pipeline.DYNAMIC_NODES[i];
            pos = obj.rect.position;
            positions[obj.id] = [pos.x, pos.y];
        }
        positions['viewport'] = [viewport.center.x, viewport.center.y, viewport.scaled];
        socket.emit('save_nodes', positions, function(response, status){
            if (status !== 200){
                console.log(response);
            }
        });
    }, 10000);

    socket.on('load_checkpoint', function(msg){ 
        var vis_data = msg.vis_data;            
        var pipeline_def = msg.pipeline;
        var blocks = pipeline_def.blocks;
        var custom_args = pipeline_def.custom_args;
        var connections = pipeline_def.connections;
        var nodes = [];
        var outs = [];
        var names = [];
        for (var i=0; i<pipeline_def.nodes.length; i++){
            nodes.push(pipeline_def.nodes[i][0])
            outs.push(pipeline_def.nodes[i][1])
            names.push(pipeline_def.nodes[i][2])
        }

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

        var node, obj, block, block_dict, arg, j, key;
        for (i=0; i<nodes.length; i++){
            // Create blocks
            block_dict = blocks[i];
            block = new Block(block_dict.name, block_dict.input_args, block_dict.custom_args,
                              block_dict.custom_args_type, block_dict.output_names,
                              block_dict.tag, block_dict.data_type);
            obj = pipeline.spawn_node_visual(block, nodes[i]);
            pos = new PIXI.Point(vis_data[nodes[i]][0], vis_data[nodes[i]][1]);
            obj.rect.position.set(pos.x, pos.y);
            node = pipeline.find_node(nodes[i]);
            node.name = names[i]
            node.is_output = outs[i];
            node.set_output(outs[i]);

            for (j=0; j<Object.keys(custom_args[i]).length; j++){
                key = Object.keys(custom_args[i])[j];
                arg = Object.values(custom_args[i])[j];
                //pipeline.set_custom_arg(obj, key, arg);
                node.block.custom_args[key] = arg;
            }
        }

        // Create actual connections
        var conn, to, from;
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
                }
                app.renderer.render(viewport)
                update_all_lines(node);
            }
        }

        if (vis_data.viewport !== undefined){
            viewport.setZoom(vis_data.viewport[2], true);
            viewport.moveCenter(vis_data.viewport[0], vis_data.viewport[1]);
        }

    });

    //socket.emit('test_receive', 'test_send_see_me_python')
});
//$('#log').html(numbers_string);

