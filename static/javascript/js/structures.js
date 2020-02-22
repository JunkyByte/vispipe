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
            this.visrect = draw_rect(VIS_RAW_SIZE + 8, Number(VIS_RAW_SIZE * 2 / 4) + 4, BLOCK_COLOR, 1);
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
        pipeline.vis[this.id + '-' + this.block.name] = this;  // TODO: If node remove is added we need to change this
        this.rect.addChild(this.visrect);
    }

    update_texture(texture) {
        this.vissprite.texture = texture;
        let ratio = VIS_IMAGE_SIZE / texture.width;
        this.vissprite.scale.set(ratio);
    }

    update_text(text) {
        var height = this.visrect.height;

        // TODO: This is temporary solution to a complex problem
        // TODO: Fallback if font becomes negative this will crash
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
            if (height <= currentHeight) {
                k += 1;
            } else {
                this.vissprite.style = style;
                this.vissprite.text = text;
                return
            }
        }

    }
}


class Block {
    constructor(name, input_args, custom_args, output_names, tag, data_type) {
        this.name = name;
        this.input_args = input_args;
        this.custom_args = custom_args;
        this.output_names = output_names;
        this.tag = tag;
        this.data_type = data_type;
    }
}

class Button {
    constructor(name, hidden=false) {
        var [width, height] = name_to_size(name);
        this.rect = draw_rect(width, height, BUTTON_COLOR, 0.8);
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

    spawn_node(block){
        socket.emit('new_node', block, function(response, status) {
            if (status === 200){
                var block = new Block(response.name, response.input_args, response.custom_args, response.output_names, response.tag, response.data_type);
                var instance;
                if (block.tag == 'vis') { // This should be passed as a bool so that is not hardcoded
                    instance = new VisNode(block, response.id);
                } else {
                    instance = new Node(block, response.id);
                }
                pipeline.DYNAMIC_NODES.push(instance);
            } else {
                console.log(response);
            }
        });
    }

    add_connection(from_block, from_idx, out_idx, to_block, to_idx, inp_idx){
        socket.emit('new_conn',
            {'from_block': from_block, 'from_idx': from_idx, 'out_idx': out_idx, 'to_block': to_block, 'to_idx': to_idx, 'inp_idx': inp_idx},
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
}


class PopupMenu {
    constructor() {
        this.flag_over = false;
        this.pane_height = 100;
        this.pane = draw_rect(CUSTOM_ARG_SIZE, this.pane_height, BLOCK_COLOR, 1);
        this.input_container = new PIXI.Container();
        this.delete_button = new Button(' DELETE ', true);
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
        var block = this.target.node.block;
        var custom_args = block.custom_args;
        
        var value, height;
        var x = CUSTOM_ARG_SIZE - 215;
        var y = 15;

        // Draw the menu
        var length = Object.keys(custom_args).length
        for (var key in custom_args) {
            if (custom_args.hasOwnProperty(key)) {
                value = custom_args[key];
                console.log(key, value); // TODO: Add type hinting
                var input_text = draw_text_input(String(value), 1);  // TODO: Add events
                var key_text = draw_text(key, 1);
                height = input_text.height;
                input_text.position.set(x, y);
                key_text.position.set(7, y + 2);
                y += height + 5;
                this.input_container.addChild(input_text);
                this.input_container.addChild(key_text);
            }
        }

        this.delete_button.rect.position.set(CUSTOM_ARG_SIZE / 2, y);
        this.input_container.addChild(this.delete_button.rect);

        this.pane.scale.y = (y + 40) / this.pane_height;
        for (var i=0; i<this.pane.children.length; i++){
            var obj = this.pane.children[i]
            obj.scale.y = 1 / this.pane.scale.y;
        }

        this.pane.interactive = true;
        this.pane.position.set(this.target.width + 5, 0);
        this.target.addChild(this.pane);
    }

    over_menu(event) {
        this.flag_over = true;
    }

    out_menu(event) {
        console.log(event);
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
        this.start_button = new Button('  RUN  ');
        this.start_button.rect.on('mousedown', ev => this.start_button.disable_button(), false);
        this.start_button.rect.on('mousedown', ev => pipeline.run_pipeline(), false);
        this.start_button.rect.position.set(0, HEIGHT - this.start_button.rect.height + 3);

        this.stop_button = new Button('  STOP  ');
        this.stop_button.rect.on('mousedown', ev => this.stop_button.disable_button(), false);
        this.stop_button.rect.on('mousedown', ev => pipeline.stop_pipeline(), false);
        this.stop_button.rect.position.set(this.start_button.rect.width - 2, HEIGHT - this.stop_button.rect.height + 3);

        this.state_text = new PIXI.Text('State: ' + pipeline.state, {fontFamily: FONT, fill: TEXT_COLOR});
        this.state_text.position.set(this.stop_button.rect.position.x + this.stop_button.rect.position.x + 20, HEIGHT - this.state_text.height);
        app.stage.addChild(this.state_text);
    }

    update_state(){
        this.state_text.text = 'State: ' + pipeline.state;
    }

    resize_menu(){
        this.start_button.rect.position.set(0, HEIGHT - this.start_button.rect.height + 3);
        this.stop_button.rect.position.set(this.start_button.rect.width - 2, HEIGHT - this.stop_button.rect.height + 3);
        this.state_text.position.set(this.stop_button.rect.position.x + this.stop_button.rect.position.x + 20, HEIGHT - this.state_text.height);
    }
}

class SideMenu {
    constructor() {
        this.tags = [];
        this.tag_button = [];
        this.pane = {};
        this.tag_idx = 0;
        this.selected_tag = null;
        for (var i = 0; i < 3; i++) {
            var button = new Button('tab-' + i.toString());
            button.rect.on('mousedown', ev => sidemenu.update_tag_blocks(ev.target.button), false);
            button.rect.position.set(WIDTH - (3 - i) * button.rect.width + 1, 0);

            this.tag_button.push(button);
        }
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

        this.update_tag_labels();
        this.selected_tag = this.tags[0];
        this.update_tag_blocks(this.tag_button[0]);

        var app_length = app.stage.children.length;
        for (var i = 0; i < this.tag_button.length; i++){
            app.stage.setChildIndex(this.tag_button[i].rect, app_length-1)
        }
    }

    update_tag_labels(){
        for (var i = 0; i < 3; i++){
            if (i < this.tags.length) {
                var name = this.tags[(i + this.tag_idx) % this.tags.length];
                this.tag_button[i].text.text = name;
            }
        }
    }

    update_tag_blocks(button){  // TODO: This logic is not connected to anything
        let tag = button.text.text;
        this.pane[this.selected_tag].visible = false;
        this.selected_tag = tag;
        this.pane[this.selected_tag].visible = true;
    }

    scroll_blocks(ev){
        for (var i = 0; i < sidemenu.tags.length; i++){
            if (sidemenu.pane[sidemenu.tags[i]].visible === true){
                var new_y = sidemenu.pane[sidemenu.tags[i]].y += ev.wheelDelta / 5;
                new_y = Math.max(new_y, -(50 - HEIGHT + 50 * sidemenu.pane[sidemenu.tags[i]].children.length));
                sidemenu.pane[sidemenu.tags[i]].y = Math.min(0, new_y);
            }
        }
    }

    resize_menu(){
        for (var i = 0; i < 3; i++) {
            this.tag_button[i].rect.position.set(WIDTH - (3 - i) * this.tag_button[i].rect.width + 1, 0);
        }

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
