import React from 'react';
import logo from './logo.svg';
import './App.css';
import * as PIXI from "pixi.js"
import * as STRUCTURES from "./structures.js"
import * as UTILS from "./utils.js"
import $ from "jquery"
import io from "socket.io-client"
import Sidebar from './Sidebar';

export let app = new PIXI.Application({
  antialias: true,
  autoResize: true,
  resolution: window.devicePixelRatio
});

app.renderer.backgroundColor = 0x202125;
app.renderer.view.style.position = 'absolute';
app.renderer.view.style.display = 'block';
document.body.appendChild(app.view);
export var WIDTH = app.renderer.width / app.renderer.resolution;
export var HEIGHT = app.renderer.height / app.renderer.resolution;
export var VIS_IMAGE_SIZE = 128;
export var VIS_RAW_SIZE = 256;
export var FONT = 'Arial';
export var CUSTOM_ARG_SIZE = 350;
export var FONT_SIZE = 18;
export var VIS_FONT_SIZE = 18;
export var TEXT_COLOR = 'white';
export var BUTTON_COLOR = 0x5DBCD2;
export var BLOCK_COLOR = 0x5DBCD2;
export var INPUT_COLOR = 0x5DBCD2;  // TODO: Add me
export var INPUT_WRONG_COLOR = 0xED1909;
export var INPUT_TEXT_COLOR = 0x26272E;
export var OUTPUT_COLOR = 0x5DBCD2;

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
export var pipeline = new STRUCTURES.Pipeline();
export var sidemenu = new STRUCTURES.SideMenu();
export var runmenu = new STRUCTURES.RunMenu();
export var popupmenu = new STRUCTURES.PopupMenu();
window.addEventListener('resize', function() {runmenu.resize_menu()}, false);
window.addEventListener('resize', function() {sidemenu.resize_menu()}, false);
window.addEventListener('mousewheel', function(ev){sidemenu.scroll_blocks(ev)}, false);

export var socket;
$(document).ready(function(){
  socket = io.connect('http://' + document.domain + ':' + window.location.port);

  socket.on('new_block', function(msg) {
      var block = new STRUCTURES.Block(msg.name, msg.input_args, msg.custom_args, msg.custom_args_type,
                            msg.output_names, msg.tag, msg.data_type);
      pipeline.STATIC_NODES.push(new STRUCTURES.StaticNode(block));
  });

  socket.on('end_block', function(msg) {
      sidemenu.populate_menu(pipeline, app);
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

  socket.on('auto_save', function(msg){
      var obj, pos, positions;
      var save_checkpoint = setInterval(() => {
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
      }, 10000);
  });

  socket.on('load_checkpoint', function(msg){
      var vis_data = msg.vis_data;
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
      var block, block_dict, arg, j;
      for (i=0; i<nodes.length; i++){
          // Create blocks
          block_dict = blocks[i];
          block = new STRUCTURES.Block(block_dict.name, block_dict.input_args, block_dict.custom_args,
                            block_dict.custom_args_type, block_dict.output_names,
                            block_dict.tag, block_dict.data_type);
          obj = pipeline.spawn_node_visual(block, nodes[i]);
          obj.rect.position.set(pos_dict[nodes[i]][0], pos_dict[nodes[i]][1]);

          for (j=0; j<Object.keys(custom_args[i]).length; j++){
              var key = Object.keys(custom_args[i])[j];
              arg = Object.values(custom_args[i])[j];
              pipeline.set_custom_arg(obj, key, arg);
          }
      }

      // Create actual connections
      var conn, to, from, to_pos, from_pos;
      for (i=0; i<pipeline.DYNAMIC_NODES.length; i++){
          var node = pipeline.DYNAMIC_NODES[i];

          conn = conn_dict[node.id];
          if (conn !== undefined) {
              for (j=0; j<conn.length; j++){
                  var to_node = pipeline.find_node(conn[j][0]);
                  from = node.out_c[conn[j][1]]
                  to = to_node.in_c[conn[j][2]]

                  conn = UTILS.create_connection(to, from); 
                  app.stage.addChildAt(conn, app.stage.children.length);

                  app.renderer.render(node.rect)  // Force rendering of the two objects to update lines correctly
                  app.renderer.render(to_node.rect)
                  UTILS.update_all_lines(node);
              }
          }

      }
  });

  //socket.emit('test_receive', 'test_send_see_me_python')
});



function App() {
  return (
    <Sidebar/>
  );
}

export default App;
