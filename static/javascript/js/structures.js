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

class Block {
    constructor(name, input_args, custom_args, output_names, tag) {
        this.name = name;
        this.input_args = input_args;
        this.custom_args = custom_args;
        this.output_names = output_names;
        this.tag = tag;
    }
}

class Button {
    constructor(name) {
        var [width, height] = name_to_size(name);
        this.rect = draw_rect(width, height, BUTTON_COLOR, 0.8);
        this.rect.buttonMode = true;
        this.text = draw_text(name, 0.9);
        this.text.anchor.set(0.5, 0.5);
        this.text.position.set(this.rect.width / 2, this.rect.height / 2);
        this.rect.addChild(this.text);
        this.rect.interactive = true;
    }
}

class Pipeline {
    // TODO: Convert to singleton (and change all reference to singleton instance)
    constructor() {
        this.STATIC_NODES = [];
        this.DYNAMIC_NODES = [];
    }

    spawn_node(block){
        socket.emit('new_node', block);
    }

    add_connection(from_block, from_idx, out_idx, to_block, to_idx, inp_idx){
        socket.emit('new_conn', {'from_block': from_block, 'from_idx': from_idx, 'out_idx': out_idx,
                                 'to_block': to_block, 'to_idx': to_idx, 'inp_idx': inp_idx}); 
    }

    remove_connection(block, id){
        console.log(block, id);
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
            button.rect.position.set(WIDTH - (3 - i) * button.rect.width + 1, 0);
            this.tag_button.push(button);
            app.stage.addChild(button.rect)
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

        this.selected_tag = this.tags[0];
        this.update_tag_labels();
        this.update_tag_blocks();

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

    update_tag_blocks(){  // TODO: This logic is not connected to anything
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
