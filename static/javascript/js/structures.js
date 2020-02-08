class AbstractNode {
    constructor(block) {
        this.block = block;
        [this.rect, this.text] = draw_block(this.block.name);
    }
}

class StaticNode extends AbstractNode {
    constructor(block) {
        super(block);
        this.rect.buttonMode = true;
        this.rect.interactive = true;
    }
}

class Node extends AbstractNode {
    constructor(block) {
        super(block);
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
        this.rect = draw_rect(width, height, color=BUTTON_COLOR, scale=0.8);
        this.rect.buttonMode = true;
        this.text = draw_text(name, scale=0.9);
        this.text.anchor.set(0.5, 0.5);
        this.text.position.set(this.rect.width / 2, this.rect.height / 2);
        this.rect.addChild(this.text);
        this.rect.interactive = true;
    }
}
class Pipeline {
    constructor() {
        this.STATIC_NODES = [];
        this.DYNAMIC_NODES = [];
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

        for (var i = 0; i < this.tags.length; i++){
            var y = 45;
            for (var j = 0; j < this.pane[this.tags[i]].children.length; j++){
                var x = WIDTH - this.pane[this.tags[i]].children[j].width;
                this.pane[this.tags[i]].children[j].position.set(x, y);
                y += 50
            }
        }

        this.selected_tag = this.tags[0];
        this.update_tag_labels();
        this.update_tag_blocks();

        var app_length = app.stage.children.length;
        for (var i = 0; i < this.tag_button.length; i++){  // TODO: Hacky but works
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

    update_tag_blocks(){
        this.pane[this.selected_tag].visible = true;
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