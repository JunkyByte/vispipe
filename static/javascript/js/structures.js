class AbstractNode {
    constructor(block) {
        this.block = block;
        [this.rect, this.text] = draw_block(this.block.name);
        this.rect.buttonMode = true;
        this.rect.interactive = true;
        this.rect.node = this;
    }
}

class StaticNode extends AbstractNode {
    constructor(block) {
        super(block);
        this.rect.on('mousedown', ev => pipeline.spawn_node(this.block), false);  // TODO: Refactor this into this.rect attribute
        this.rect.on('touchstart', ev => pipeline.spawn_node(this.block), false);
    }
}

class Node extends AbstractNode {
    constructor(block, id) {
        super(block);
        this.id = id;
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
            .on('touchmove', onDragMove)
            .on('mouseover', onMouseOver)
            .on('mouseout', onMouseOut);

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
        this.rect.position.set(WIDTH/3, HEIGHT/2);
        app.stage.addChild(this.rect);
    }
}

class VisNode extends Node {
    constructor(block, id) {
        super(block, id);

        function setpos(rect, visrect){
            var delta = rect.width - visrect.width;
            visrect.position.set(delta / 2, 40);
        }

        if (this.block.data_type === 'raw'){
            this.visrect = draw_rect(VIS_RAW_SIZE + 8, Number(VIS_RAW_SIZE * 1 / 4) + 4, BLOCK_COLOR, 1);
            setpos(this.rect, this.visrect);
            this.vissprite = new PIXI.Text('', new PIXI.TextStyle());
        } else if (this.block.data_type === 'image'){
            this.visrect = draw_rect(VIS_IMAGE_SIZE + 8, VIS_IMAGE_SIZE + 8, BLOCK_COLOR, 1);
            setpos(this.rect, this.visrect);

            let texture = PIXI.Texture.EMPTY  // Temporary texture
            this.vissprite = PIXI.Sprite.from(texture);
            this.update_texture(texture);

        }
        if (this.vissprite !== undefined) {
            this.visrect.addChild(this.vissprite)
            this.vissprite.position.set(4, 4);
        }
        pipeline.vis[this.id] = this;
        this.rect.addChild(this.visrect);
    }

    update_texture(texture) {
        this.vissprite.texture = texture;
        let ratio = VIS_IMAGE_SIZE / texture.width;
        this.vissprite.scale.set(ratio);
    }

    update_text(text) {
        var height = this.visrect.height;

        var k = 0;
        var style;
        while (true) {
            style = new PIXI.TextStyle({
                fontFamily: FONT,
                breakWords: true,
                fontSize: VIS_FONT_SIZE - k,
                wordWrap: true,
                align: 'left',
                fill: TEXT_COLOR,
                wordWrapWidth: this.visrect.width - 8,
            });

            var currentHeight = PIXI.TextMetrics.measureText(text, style).height;
            if (height <= currentHeight && VIS_FONT_SIZE - k !== 1) {
                k += 1;
            } else {
                this.vissprite.style = style;
                this.vissprite.text = text;
                break
            }
        }

    }
}


class Block {
    constructor(name, input_args, custom_args, custom_args_type, output_names, tag, data_type) {
        this.name = name;
        this.input_args = input_args;
        this.custom_args = custom_args;
        this.custom_args_type = custom_args_type;
        this.output_names = output_names;
        this.tag = tag;
        this.data_type = data_type;
    }
}

class Button {
    constructor(name, hidden=false) {
        var [width, height] = name_to_size(name);
        this.rect = draw_rect(width, height, BUTTON_COLOR, 0.8);
        this.rect.on('mouseover', ev => { this.rect.alpha = 0.9 }, false);
        this.rect.on('mouseout', ev => { this.rect.alpha = 1 }, false);
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
        this.DYNAMIC_NODES = [];
        this.vis = {};
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
                    block = closure;
                    block = new Block(block.name, block.input_args, block.custom_args,
                                      block.custom_args_type, block.output_names,
                                      block.tag, block.data_type);
                    self.spawn_node_visual(block, response.id)
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
                    app.stage.removeChild(node_closure.rect);

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
                node.block.custom_args[key] = value;
            } else {
                console.log(response);
            }
        });
    }

    add_connection(from_hash, out_idx, to_hash, inp_idx){
        socket.emit('new_conn',
            {'from_hash': from_hash, 'out_idx': out_idx, 'to_hash': to_hash, 'inp_idx': inp_idx},
            function(response, status){
                if (status !== 200){
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

                var in_c, obj;
                for (var i=0; i<pipeline.DYNAMIC_NODES.length; i++){
                    app.stage.removeChild(pipeline.DYNAMIC_NODES[i].rect);
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
        this.delete_button.rect.on('mousedown', ev => pipeline.remove_node(this.currentNode), false);
        this.pane.addChild(this.input_container);
        this.pane.buttonMode = true;
        this.pane.interactive = false;
        this.pane.on('mouseover', ev => this.over_menu(ev), false);
        this.pane.on('mouseout', ev => this.out_menu(ev), false);
    }

    show_menu(ev) {
        if (ev.target === undefined){
            return;
        }

        if (this.pane.parent) {  // TODO: Can be refactored
            this.flag_over = true;
            this.out_menu();
        }

        this.target = ev.target;
        this.currentNode = this.target.node;
        var block = this.target.node.block;
        var custom_args = block.custom_args;
        var custom_args_type = block.custom_args_type;
        
        var value, type, height;
        var x = CUSTOM_ARG_SIZE - 215;
        var y = 15;

        var self = this;

        // Draw the menu
        var length = Object.keys(custom_args).length
        for (var key in custom_args) {
            if (custom_args.hasOwnProperty(key)) {
                value = custom_args[key];
                type = custom_args_type[key];
                var input_text = draw_text_input(String(value), 1);
                input_text.text = String(value);

                if (type === 'int'){
                    input_text.restrict = '0123456789';
                } else if (type === 'float'){
                    input_text.restrict = '0123456789.';
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
                key_text.position.set(7, y + 2);
                y += height + 5;
                this.input_container.addChild(input_text);
                this.input_container.addChild(key_text);
            }
        }

        var scale_y, scale_x, delete_pos;
        if (length === 0){
            scale_x = this.delete_button.rect.width / CUSTOM_ARG_SIZE;
            scale_y = this.delete_button.rect.height / this.pane_height;
            delete_pos = new PIXI.Point(1, 0.5)
        } else {
            scale_x = 1
            scale_y = (y + 40) / this.pane_height;
            delete_pos = new PIXI.Point(CUSTOM_ARG_SIZE / 2, y - 2)
        }

        this.delete_button.rect.position.set(delete_pos.x, delete_pos.y);
        this.input_container.addChild(this.delete_button.rect);

        this.pane.scale.set(scale_x, scale_y)
        for (var i=0; i<this.pane.children.length; i++){
            var obj = this.pane.children[i]
            obj.scale.y = 1 / this.pane.scale.y;
            obj.scale.x = 1 / this.pane.scale.x;
        }

        this.pane.interactive = true;
        this.pane.position.set(this.target.geometry.bounds.maxX + 5, 0);
        this.target.addChild(this.pane);
    }

    over_menu(event) {
        this.flag_over = true;
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
            this.flag_over = false;
            this.input_container.destroy();
            this.input_container = new PIXI.Container();
            this.pane.scale.y = 1;
            this.pane.addChild(this.input_container);
            this.target.removeChild(this.pane);
            this.target = undefined;
        }
    }
}

class RunMenu {
    constructor() {
        var x = 0;
        this.start_button = new Button('  RUN  ');
        this.start_button.rect.on('mousedown', ev => this.start_button.disable_button(), false);
        this.start_button.rect.on('mousedown', ev => pipeline.run_pipeline(), false);
        this.start_button.rect.position.set(x, HEIGHT - this.start_button.rect.height + 3);
        x += this.start_button.rect.width - 2

        this.stop_button = new Button('  STOP  ');
        this.stop_button.rect.on('mousedown', ev => this.stop_button.disable_button(), false);
        this.stop_button.rect.on('mousedown', ev => pipeline.stop_pipeline(), false);
        this.stop_button.rect.position.set(x, HEIGHT - this.stop_button.rect.height + 3);
        x += this.stop_button.rect.width - 2

        this.clear_button = new Button('  CLEAR  ');
        this.clear_button.rect.on('mousedown', ev => this.clear_button.disable_button(), false);
        this.clear_button.rect.on('mousedown', ev => pipeline.clear_pipeline(), false);
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
        this.next_button.rect.on('mousedown', ev => { sidemenu.scroll_tag(true) });

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
        this.prev_button.rect.on('mousedown', ev => { sidemenu.scroll_tag(false) });
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
                var x = WIDTH - this.pane[this.tags[i]].children[j].width;
                this.pane[this.tags[i]].children[j].position.set(x, y);
                y += 50
            }
        }

        var index = Object.keys(this.pane).indexOf('None')
        this.tag_idx = 0

        this.update_tag_labels();
        this.selected_tag = this.tags[0];
        this.update_tag_blocks(this.tag_button[0]);

        var app_length = app.stage.children.length;
        for (var i = 0; i < this.tag_button.length; i++){
            app.stage.setChildIndex(this.tag_button[i].rect, app_length-1);
        }
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
