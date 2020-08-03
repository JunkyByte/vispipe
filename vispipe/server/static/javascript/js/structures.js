class AbstractNode {
    constructor(block) {
        this.block = block;
        [this.rect, this.text] = draw_block(this.block.name);
        this.width = this.rect.width;
        this.height = this.rect.height;
        this.rect.buttonMode = true;
        this.rect.interactive = true;
        this.rect.node = this;
        this.rect
            .on('mouseover', onMouseOver)
            .on('mouseout', onMouseOut);
    }
}

class StaticNode extends AbstractNode {
    constructor(block) {
        super(block);
        this.rect.on('mousedown', _ => pipeline.spawn_node(this.block), false);  // TODO: Refactor this into this.rect attribute
        this.rect.on('touchstart', _ => pipeline.spawn_node(this.block), false);

        if (this.block.docstring !== null){
            var doc_width = 256;
            var style = new PIXI.TextStyle({
                fontFamily: FONT,
                breakWords: true,
                fontSize: VIS_FONT_SIZE,
                wordWrap: true,
                align: 'left',
                fill: TEXT_COLOR,
                wordWrapWidth: doc_width - 8,
            });

            this.doc_block = draw_rect(doc_width, PIXI.TextMetrics.measureText(this.block.docstring, style).height + 15, BLOCK_COLOR, 1);
            this.doc_text = new PIXI.Text(this.block.docstring, style);
            this.doc_block.addChild(this.doc_text);
            this.doc_text.position.set(6, 4);
            this.doc_block.position.set(- doc_width - 10, 0);
            this.doc_block.visible = false;
            this.rect.addChild(this.doc_block);
            this.rect
                .on('mouseover', _ => {
                    this.doc_block.visible = true;
                    this.rect.parent.setChildIndex(this.rect, this.rect.parent.children.length - 1)
                }, false)
                .on('mouseout', _ => { this.doc_block.visible = false }, false);
        }
    }
}

class Node extends AbstractNode {
    constructor(block, id) {
        super(block);
        this.id = id;
        this.name = null;
        this.is_output = false;
        this.is_macro = false;
        this.rect
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

        [this.in_c, this.out_c] = draw_conn(Object.keys(block.input_args).length, block.output_names.length, this.rect)
        for (var i=0; i<this.in_c.length; i++){
            this.in_c[i].index = i;
            this.in_c[i].type = 'input';
            this.in_c[i].connection = null;
            this.in_c[i].conn_line = null;
            this.rect.addChild(this.in_c[i]);
        }
        for (i=0; i<this.out_c.length; i++){
            this.out_c[i].index = i;
            this.out_c[i].type = 'output';
            this.out_c[i].connection = [];
            this.out_c[i].conn_line = [];
            this.rect.addChild(this.out_c[i]);
        }
        this.rect.position.set(viewport.center.x, viewport.center.y);
        viewport.addChild(this.rect);
    }

    set_output(){
        this.is_output = !this.is_output;
        this.update_color();
    }

    set_macro(value){
        this.is_macro = !this.is_macro;
        this.update_color();
    }

    update_color(){
        var color = BLOCK_COLOR;
        if (this.is_output && this.is_macro){
            color = BLOCK_BOTH_COLOR;
        } else if (this.is_output) {
            color = BLOCK_OUT_COLOR;
        } else if (this.is_macro) {
            color = BLOCK_MACRO_COLOR;
        }

        this.rect.clear();
        var [width, height] = name_to_size(this.text.text);
        this.rect = draw_rect(width, height, color, 1, this.rect);
    }
}

class VisNode extends Node {
    constructor(block, id) {
        super(block, id);

        if (this.block.data_type === 'raw'){
            this.visrect = draw_rect(VIS_RAW_SIZE + 8, Number(VIS_RAW_SIZE * 1 / 4) + 4, BLOCK_COLOR, 1);
            this.setpos(this.rect, this.visrect);
            this.vissprite = new PIXI.Text('', new PIXI.TextStyle());
        } else if (this.block.data_type === 'image' || this.block.data_type === 'plot'){
            this.visrect = draw_rect(VIS_IMAGE_SIZE + 8, VIS_IMAGE_SIZE + 8, BLOCK_COLOR, 1);
            this.setpos(this.rect, this.visrect);

            let texture = PIXI.Texture.EMPTY  // Temporary texture
            this.vissprite = PIXI.Sprite.from(texture);
        }

        this.visrect.addChild(this.vissprite)
        this.vissprite.position.set(4, 4);
        pipeline.vis[this.id] = this;
        this.rect.addChild(this.visrect);
    }

    setpos(rect, visrect){
        var delta = rect.geometry.bounds.maxX - visrect.width;
        visrect.position.set(delta / 2, 40);
    }

    update_texture(texture) {
        this.vissprite.texture = texture;
        if (texture.width >= texture.height){
            var ratio = texture.width / texture.height;
            this.vissprite.scale.x = ratio * VIS_IMAGE_SIZE / texture.width;
            this.vissprite.scale.y = VIS_IMAGE_SIZE / texture.height;
        } else {
            var ratio = texture.height / texture.width;
            this.vissprite.scale.x = VIS_IMAGE_SIZE / texture.width;
            this.vissprite.scale.y = ratio * VIS_IMAGE_SIZE / texture.height;
        }
        this.visrect.scale.x = (texture.width * this.vissprite.scale.x) / VIS_IMAGE_SIZE;
        this.visrect.scale.y = (texture.height * this.vissprite.scale.y) / VIS_IMAGE_SIZE;
        this.vissprite.scale.x = this.vissprite.scale.x / this.visrect.scale.x;
        this.vissprite.scale.y = this.vissprite.scale.y / this.visrect.scale.y;
        this.setpos(this.rect, this.visrect);
    }

    update_text(text) {
        var height = this.visrect.height;
        var currentHeight = Infinity;

        var style = new PIXI.TextStyle({
            fontFamily: FONT,
            breakWords: true,
            fontSize: VIS_FONT_SIZE * 2,
            wordWrap: true,
            align: 'left' - 10,
            fill: TEXT_COLOR,
            wordWrapWidth: this.visrect.width - 8,
        });
        while (currentHeight >= (height - 8) && style.fontSize !== 1) {
            style.fontSize -= 1
            currentHeight = PIXI.TextMetrics.measureText(text, style).height;
        }
        this.vissprite.style = style;
        this.vissprite.text = text;

    }
}


class Block {
    constructor(name, input_args, input_args_type, output_names, tag, data_type, docstring) {
        this.name = name;
        this.input_args = input_args;
        this.input_args_type = input_args_type;
        this.output_names = output_names;
        this.tag = tag;
        this.data_type = data_type;
        this.docstring = docstring;
        if (this.docstring !== null){
            this.docstring = this.docstring.replace('    ', '')
        }
    }
}

class Button {
    constructor(name, hidden=false) {
        var [width, height] = name_to_size(name);
        this.rect = draw_rect(width, height, BUTTON_COLOR, 0.8);
        this.rect.on('mouseover', _ => { this.rect.alpha = 0.9 }, false);
        this.rect.on('mouseout', _ => { this.rect.alpha = 1 }, false);
        this.text = draw_text(name, 0.9);
        this.text.anchor.set(0.5, 0.5);
        this.text.position.set(this.rect.width / 2, this.rect.height / 2);
        this.rect.addChild(this.text);
        this.rect.interactive = true;
        this.rect.buttonMode = true;
        this.rect.button = this;
        if (! hidden) {
            app.stage.addChild(this.rect)
        }
    }

    disable_button(){
        this.rect.interactive = false;
    }

    enable_button(){
        this.rect.interactive = true;
    }
}


const RunState = {
    IDLE: 'idle',
    RUNNING: 'running',
}


class Pipeline {
    // TODO: Convert to singleton (and change all reference to singleton instance)
    constructor() {
        this.state = RunState.IDLE;
        this.STATIC_NODES = [];
        this.NAMES = [];
        this.DYNAMIC_NODES = [];
        this.vis = {};
        this.current_macro = null;
    }

    find_node(id){
        for (var i=0; i<this.DYNAMIC_NODES.length; i++){
            if (this.DYNAMIC_NODES[i].id == id){
                return this.DYNAMIC_NODES[i];
            }
        }
        return undefined
    }

    spawn_node(block){
        var self = this;
        socket.emit('new_node', block, function() {
            var closure = block;  // Creates closure for block
            return function(response, status) {
                if (status === 200){
                    let newblock = $.extend(true,{}, closure);
                    self.spawn_node_visual(newblock, response.id);
                } else {
                    console.log(response);
                }
            }
        }());
    }
    
    spawn_node_visual(block, id){
        var instance;
        if (block.tag == 'vis') { // This should be passed as a bool so that is not hardcoded
            instance = new VisNode(block, id);
        } else {
            instance = new Node(block, id);
        }
        this.DYNAMIC_NODES.push(instance);
        return instance;
    }

    remove_node(node){
        var self = this;
        socket.emit('remove_node', node.id, function() { // I think this can be simplified by creating node esplicitly
            var node_closure = node;
            return function(reponse, status) {
                if (status === 200){
                    var index = self.DYNAMIC_NODES.indexOf(node_closure);
                    node_closure = self.DYNAMIC_NODES.splice(index, 1)[0];
                    for (var i=0; i<node_closure.in_c.length; i++){
                        clear_connection(node_closure.in_c[i]);
                    }

                    var j, out;
                    for (i=0; i<node_closure.out_c.length; i++){
                        out = node_closure.out_c[i]; 
                        for (j=out.connection.length - 1; j>=0; j--){
                            clear_connection(out.connection[j]);
                        }
                    }
                    viewport.removeChild(node_closure.rect);

                    delete pipeline.vis[node_closure.id];
                } else {
                    console.log(response);
                }
            }
        }());
    }

    set_custom_arg(node, key, value){
        socket.emit('set_custom_arg', {'id': node.id, 'key': key, 'value': value}, function(response, status){
            if (status === 200){
                console.log('The custom arg has been set');
                node.block.input_args[key] = value;
            } else {
                console.log(response);
            }
        });
    }

    set_name(node, name){
        socket.emit('set_name', {'id': node.id, 'name': name}, function(){
            var node_closure = node;
            var name_closure = name;
            return function(response, status){
                if (status === 200){
                    console.log('The name of the node has been set');
                    node_closure.name = name_closure;
                } else {
                    console.log(response);
                }
            }
        }());
    }

    set_output(node){
        socket.emit('set_output', {'id': node.id, 'state': !node.is_output}, function() {
            var node_closure = node;
            return function(response, status){
                if (status === 200){
                    node_closure.set_output()
                } else {
                    console.log(response);
                }
            }
        }());
    }
    
    set_macro(node){
        if (node.is_macro){
            this.current_macro = node;
        }

        if (this.current_macro === null){
            this.current_macro = node;
            return;
        }

        var self = this;
        socket.emit('set_macro', {'from_id': this.current_macro.id, 'to_id': node.id, 'state': !this.current_macro.is_macro},
            function(response, status){
                if (status === 200){
                    var edges = response.edges;
                    var node;
                    for (var i=0; i<edges.length; i++){
                        node = self.find_node(edges[i]).set_macro();
                    }
                } else {
                    console.log(response);
                }
            });

        this.current_macro = null;
    }

    add_connection(output_node, output, input_node, input, start_pos, end_pos){
        socket.emit('new_conn',
            {
                'from_hash': output_node.id,
                'out_idx': output.index,
                'to_hash': input_node.id,
                'inp_idx': input.index
            },
            function(response, status){
                if (status === 200){
                    // If we connect a input node which is already connected we need to remap
                    // its connection to the new output
                    // If we connect a output node which is already connected we need to APPEND
                    // its new connection
                    var line = create_connection(input, output);  // Create the visual connection
                    viewport.addChildAt(line, viewport.children.length);
                    update_line(line, start_pos, end_pos);
                } else {
                    console.log(response);
                }
            }); 
    }

    run_pipeline(){
        var self = this;
        socket.emit('run_pipeline', function(response, status) {
            if (status === 200) {
                self.state = RunState.RUNNING;
                runmenu.update_state();
            } else {
                console.log(response);
            }
            runmenu.start_button.enable_button();
        });
    }

    stop_pipeline(){
        var self = this;
        socket.emit('stop_pipeline', function(response, status) {
            if (status === 200) {
                self.state = RunState.IDLE;
                runmenu.update_state();
            } else {
                console.log(response);
            }
            runmenu.stop_button.enable_button();
        });
    }

    clear_pipeline(){
        var self = this;
        socket.emit('clear_pipeline', function(response, status) {
            if (status === 200) {
                self.state = RunState.IDLE;
                runmenu.update_state();

                var in_c;
                for (var i=0; i<pipeline.DYNAMIC_NODES.length; i++){
                    viewport.removeChild(pipeline.DYNAMIC_NODES[i].rect);
                    in_c = pipeline.DYNAMIC_NODES[i].in_c;
                    for (var j=0; j<in_c.length; j++){
                        clear_connection(in_c[j]);
                    }
                }
                pipeline.DYNAMIC_NODES = [];
            } else {
                console.log(response);
            }
            runmenu.clear_button.enable_button();
        });
    }
}


class PopupMenu {
    constructor() {
        this.flag_over = false;
        this.currentNode = null;
        this.pane_height = 100;
        this.pane = draw_rect(CUSTOM_ARG_SIZE, this.pane_height, BLOCK_COLOR, 1);
        this.input_container = new PIXI.Container();
        this.delete_button = new Button(' DELETE ', true);
        this.delete_button.rect.on('mousedown', _ => pipeline.remove_node(this.currentNode), false);
        this.out_button = new Button(' OUTPUT ', true);
        this.out_button.rect.on('mousedown', _ => pipeline.set_output(this.currentNode), false);
        this.macro_button = new Button('  MACRO  ', true);
        this.macro_button.rect.on('mousedown', _ => pipeline.set_macro(this.currentNode), false);
        this.pane.addChild(this.input_container);
        this.pane.buttonMode = true;
        this.pane.interactive = false;
        this.pane.on('mouseover', _ => this.over_menu(), false);
        this.pane.on('mouseout', ev => this.out_menu(ev), false);
    }

    show_menu(ev) {
        if (ev.target === undefined){
            return;
        }

        clearTimeout(this.__out_event);
        if (this.target){
            this.close_menu();
            this.flag_over = false;
        }

        this.target = ev.target;
        this.currentNode = this.target.node;
        var block = this.target.node.block;
        var custom_args = block.input_args; // TODO
        var custom_args_type = block.input_args_type;
        var value, type, height;
        var x = CUSTOM_ARG_SIZE - 215;
        var y = 15;
        var self = this;

        // Draw text field
        var name = this.currentNode.name === null ? '' : String(this.currentNode.name);
        var input_text = draw_text_input(String(name), 1);
        input_text.text = String(name);

        input_text.on('input', function(input_text) {
            return function() {
                if (String(input_text.text) !== input_text.placeholder){
                    pipeline.set_name(self.currentNode, input_text.text);
                }
            }
        }(input_text), false);
        var key_text = draw_text('node name', 1);
        height = input_text.height;
        input_text.position.set(x, y);
        key_text.position.set(7, y + 5);
        y += height + 5;
        this.input_container.addChild(input_text);
        this.input_container.addChild(key_text);

        // Draw the menu
        for (var key in custom_args) {
            if (custom_args.hasOwnProperty(key)) {
                value = custom_args[key];
                type = custom_args_type[key];
                var input_text = draw_text_input(String(value), 1);
                input_text.text = String(value);

                if (type === 'int'){
                    input_text.restrict = '-0123456789';
                } else if (type === 'float'){
                    input_text.restrict = '-0123456789.';
                }

                input_text.on('input', function(input_text, key) {
                    return function() {
                        if (input_text.text && String(input_text.text) !== input_text.placeholder){
                            pipeline.set_custom_arg(self.currentNode, key, input_text.text);
                        }
                    }
                }(input_text, key), false);
                var key_text = draw_text(key + ': ' + type, 1);
                height = input_text.height;
                input_text.position.set(x, y);
                key_text.position.set(7, y + height / 8);
                y += height + 5;
                this.input_container.addChild(input_text);
                this.input_container.addChild(key_text);
            }
        }

        var scale_y = (y + 40) / this.pane_height;
        var delete_pos = new PIXI.Point(CUSTOM_ARG_SIZE / 8, y - 2);
        var out_pos = new PIXI.Point(3 * CUSTOM_ARG_SIZE / 8, y - 2);
        var macro_pos = new PIXI.Point(5 * CUSTOM_ARG_SIZE / 8, y - 2);

        this.delete_button.rect.position.set(delete_pos.x, delete_pos.y);
        this.out_button.rect.position.set(out_pos.x, out_pos.y);
        this.macro_button.rect.position.set(macro_pos.x, macro_pos.y);
        this.input_container.addChild(this.delete_button.rect);
        this.input_container.addChild(this.out_button.rect);
        this.input_container.addChild(this.macro_button.rect);

        this.pane.scale.set(1, scale_y)
        for (var i=0; i<this.pane.children.length; i++){
            var obj = this.pane.children[i]
            obj.scale.y = 1 / this.pane.scale.y;
            obj.scale.x = 1 / this.pane.scale.x;
        }

        this.pane.interactive = true;
        this.pane.position.set(this.target.geometry.bounds.maxX + 5, 0);
        this.target.addChild(this.pane);
    }

    over_menu() {
        this.flag_over = true;
        clearTimeout(this.__out_event);
    }

    out_menu(event) {
        // Check if is a child of the pane, in that case do not close
        if (event && event.data) {
            var hit_obj = app.renderer.plugins.interaction.hitTest(event.data.global);
        }

        if (hit_obj) {
            for (var i=0; i<event.currentTarget.children.length; i++){
                var child = event.currentTarget.children[i];
                if(child === hit_obj.parent.parent){
                    return;
                }
            }
        }

        if (this.flag_over) {
            this.__out_event = setTimeout(() => {
                this.close_menu();
            }, 300);
        }
    }

    close_menu(){
        this.flag_over = false;
        this.input_container.destroy();
        this.input_container = new PIXI.Container();
        this.pane.scale.y = 1;
        this.pane.addChild(this.input_container);
        this.target.removeChild(this.pane);
        this.target = undefined;
    }
}

class RunMenu {
    constructor() {
        var x = 0;
        this.start_button = new Button('  RUN  ');
        this.start_button.rect.on('mousedown', _ => this.start_button.disable_button(), false);
        this.start_button.rect.on('mousedown', _ => pipeline.run_pipeline(), false);
        this.start_button.rect.position.set(x, HEIGHT - this.start_button.rect.height + 3);
        x += this.start_button.rect.width - 2

        this.stop_button = new Button('  STOP  ');
        this.stop_button.rect.on('mousedown', _ => this.stop_button.disable_button(), false);
        this.stop_button.rect.on('mousedown', _ => pipeline.stop_pipeline(), false);
        this.stop_button.rect.position.set(x, HEIGHT - this.stop_button.rect.height + 3);
        x += this.stop_button.rect.width - 2

        this.clear_button = new Button('  CLEAR  ');
        this.clear_button.rect.on('mousedown', _ => this.clear_button.disable_button(), false);
        this.clear_button.rect.on('mousedown', _ => pipeline.clear_pipeline(), false);
        this.clear_button.rect.position.set(x, HEIGHT - this.clear_button.rect.height + 3);
        x += this.clear_button.rect.width - 2

        this.state_text = new PIXI.Text('State: ' + pipeline.state, {fontFamily: FONT, fill: TEXT_COLOR});
        this.state_text.position.set(x + 20, HEIGHT - this.state_text.height);
        app.stage.addChild(this.state_text);
    }

    update_state(){
        this.state_text.text = 'State: ' + pipeline.state;
    }

    resize_menu(){
        var x = 0;
        this.start_button.rect.position.set(x, HEIGHT - this.start_button.rect.height + 3);
        x += this.start_button.rect.width - 2

        this.stop_button.rect.position.set(x, HEIGHT - this.stop_button.rect.height + 3);
        x += this.stop_button.rect.width - 2

        this.clear_button.rect.position.set(x, HEIGHT - this.clear_button.rect.height + 3);
        x += this.clear_button.rect.width - 2

        this.state_text.position.set(x + 20, HEIGHT - this.state_text.height);
    }
}

class SideMenu {
    constructor() {
        this.tags = [];
        this.tag_button = [];
        this.pane = {};
        this.tag_idx = 0;
        this.selected_tag = null;
        this.next_button = new Button('>>');
        this.next_button.rect.on('mousedown', function() { sidemenu.scroll_tag(false) });

        var x = WIDTH - this.next_button.rect.width
        this.next_button.rect.position.set(x, 0);
        for (var i = 0; i < 3; i++) {
            var button = new Button('tab-' + i.toString());
            button.rect.on('mousedown', ev => sidemenu.update_tag_blocks(ev.target.button), false);
            x -= button.rect.width
            button.rect.position.set(x, 0);
            this.tag_button.push(button);
        }
        this.prev_button = new Button('<<');
        this.prev_button.rect.on('mousedown', function() { sidemenu.scroll_tag(true) });
        x -= this.prev_button.rect.width
        this.prev_button.rect.position.set(x, 0);
    }

    populate_menu(pipeline){
        for (var i = 0; i < pipeline.STATIC_NODES.length; i++) {
            var tag = pipeline.STATIC_NODES[i].block.tag;
            if (this.tags.indexOf(tag) === -1) {
                this.tags.push(tag);
                this.pane[tag] = new PIXI.Container();
                app.stage.addChild(this.pane[tag]);
            }
            this.pane[tag].visible = false;
            this.pane[tag].addChild(pipeline.STATIC_NODES[i].rect);
            this.pane[tag].interactive = true;
        }

        for (i = 0; i < this.tags.length; i++){
            var y = 45;
            for (var j = 0; j < this.pane[this.tags[i]].children.length; j++){
                this.pane[this.tags[i]].children[j].scale.set(0.8);
                var obj = this.pane[this.tags[i]].children[j].node
                var x = WIDTH - obj.width * obj.rect.scale.x;
                this.pane[this.tags[i]].children[j].position.set(x, y);
                y += 50
            }
        }

        // var index = Object.keys(this.pane).indexOf('None') # TODO: Select by default?
        this.tag_idx = 0

        this.update_tag_labels();
        this.selected_tag = this.tags[0];
        this.update_tag_blocks(this.tag_button[0]);

        var app_length = app.stage.children.length;
        for (var i = 0; i < this.tag_button.length; i++){
            app.stage.setChildIndex(this.tag_button[i].rect, app_length-1);
        }
        app.stage.setChildIndex(this.next_button.rect, app_length-1);
        app.stage.setChildIndex(this.prev_button.rect, app_length-1);
    }

    scroll_tag(right){
        var delta = (right) ? 1 : -1;
        this.tag_idx = (this.tag_idx + delta) % this.tags.length;
        this.update_tag_labels();
    }

    update_tag_labels(){
        var idx;
        for (var i = 0; i < 3; i++){
            if (i < this.tags.length) {
                idx = (i + this.tag_idx) % this.tags.length;
                if (idx < 0){
                    idx = idx + this.tags.length;
                }
                var name = this.tags[idx];
                this.tag_button[i].text.text = name;
            }
        }
    }

    update_tag_blocks(button){
        let tag = button.text.text;
        this.pane[this.selected_tag].visible = false;
        this.selected_tag = tag;
        this.pane[this.selected_tag].visible = true;
    }

    scroll_blocks(ev){
        for (var i = 0; i < sidemenu.tags.length; i++){
            if (sidemenu.pane[sidemenu.tags[i]].visible === true){
                var new_y = sidemenu.pane[sidemenu.tags[i]].y += ev.deltaY / 5;
                new_y = Math.max(new_y, -(50 - HEIGHT + 50 * sidemenu.pane[sidemenu.tags[i]].children.length));
                sidemenu.pane[sidemenu.tags[i]].y = Math.min(0, new_y);
                break
            }
        }
    }

    resize_menu(){
        var x = WIDTH - this.next_button.rect.width
        this.next_button.rect.position.set(x, 0);
        for (var i = 0; i < 3; i++) {
            x -= this.tag_button[i].rect.width
            this.tag_button[i].rect.position.set(x, 0);
        }
        x -= this.prev_button.rect.width
        this.prev_button.rect.position.set(x, 0);

        for (var i = 0; i < this.tags.length; i++){
            var y = 45;
            for (var j = 0; j < this.pane[this.tags[i]].children.length; j++){
                var x = WIDTH - this.pane[this.tags[i]].children[j].width;
                this.pane[this.tags[i]].children[j].position.set(x, y);
                y += 50
            }
        }
    }
}
