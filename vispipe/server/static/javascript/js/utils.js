function onDragStart(event)  // TODO: This function can be refactored with better logic
{
    var i;
    var obj;

    this.target = event.target;
    this.ischild = false;
    for (i=0; i<event.target.children.length; i++){
        this.child = event.target.children[i];
        if (this.child.type !== undefined && this.child.containsPoint(event.data.global)) {
            obj = new PIXI.Graphics();
            this.ischild = obj;
            viewport.addChildAt(this.ischild, viewport.children.length);
            break;
        }
    }

    if (this._clicked && !this.ischild && this.target.node !== undefined) {
        console.log('is a double click');
        this.alpha = 0.5;
        popupmenu.show_menu(event);
    }
    this._clicked = false;
    clearTimeout(this.__double);

    this.start_pos = viewport.toWorld(event.data.global);
    this.data = event.data;

    if (this.target.node !== undefined){
        this.dragging = this.data.getLocalPosition(this.parent);
    }

    if (!this.ischild && this.target.node !== undefined){
        this.alpha = 0.5;
        if (viewport.children.length > 0){
            viewport.setChildIndex(this, viewport.children.length-1);
        }
    } else {
        if (viewport.children.length > 1){
            viewport.setChildIndex(this, viewport.children.length-2);
        }
    }
}

function onDragEnd(event)
{
    this._clicked = true;
    this.__double = setTimeout(() => { this._clicked = false; }, 150);

    // Close drag logic
    this.alpha = 1;
    this.dragging = false;
    // set the interaction data to null
    this.data = null;

    if (this.ischild){
        var target_conn = point_to_conn(viewport.toWorld(event.data.global)); // The connection we arrived to
        if (target_conn && target_conn.type !== this.child.type){
            var input = (this.child.type === 'input') ? this.child : target_conn;
            var output = (this.child.type === 'output') ? this.child : target_conn;
            var input_node = input.parent.node;
            var output_node = output.parent.node;
            // If we connect a input node which is already connected we need to remap
            // its connection to the new output
            // If we connect a output node which is already connected we need to APPEND
            // its new connection
            var line = create_connection(input, output)  // Create the visual connection
            viewport.addChildAt(line, viewport.children.length);
            update_line(line, this.start_pos, viewport.toWorld(event.data.global));
            viewport.removeChild(this.ischild)  // Delete temp line
            pipeline.add_connection(output_node.id, output.index,  // TODO: This is not checked server side but client side
                                    input_node.id, input.index);
        } else {
            this.ischild.destroy();
        }
    }

    this.ischild = false;
    this.start_pos = null;
    this.child = null;
    this.target = null;
}

function create_connection(input, output){
    var obj = new PIXI.Graphics();
    clear_connection(input);
    input.connection = output;
    output.connection.push(input);
    obj.from = input;
    obj.to = output;
    input.conn_line = obj;
    output.conn_line.push(obj);
    return obj
}

function clear_connection(input){
    if (input.connection){
        // Remove connection and conn_line from output (that is stored in .connection)
        var index = input.connection.connection.indexOf(input)
        input.connection.connection.splice(index, 1);
        index = input.connection.conn_line.indexOf(input.conn_line)
        input.connection.conn_line.splice(index, 1);

        // Remove connection and conn_line from input
        input.conn_line.destroy();
        input.conn_line = null;
        input.connection = null;
    }
}

function point_to_conn(point){
    var i, child;
    var root_obj = app.renderer.plugins.interaction.hitTest(point);

    if (root_obj){
        for (i=0; i<root_obj.children.length; i++){
            child = root_obj.children[i];
            if (child.type !== undefined && child.containsPoint(point)) {
                return child;
            }
        }
    }
    return null;
}

function onDragMove(event)
{
    if (this.dragging && !this.ischild)
    {
        var newPosition = this.data.getLocalPosition(this.parent);
        this.position.x += (newPosition.x - this.dragging.x);
        this.position.y += (newPosition.y - this.dragging.y);
        this.dragging = newPosition;
        update_all_lines(this.target.node);
    } else if (this.ischild) {
        update_line(this.ischild, this.start_pos, viewport.toWorld(event.data.global));
    }
}

function onMouseOver(event){
    if (!this.dragging){
        event.target.alpha = 0.9;
    }
}

function onMouseOut(event){
    if (!this.dragging){
        event.currentTarget.alpha = 1;
    }
}

function update_all_lines(node){
    var i, j, from, to, from_pos, to_pos;

    for (i=0; i<node.in_c.length; i++){
        if (node.in_c[i].conn_line){
            from = node.in_c[i].conn_line.from;
            from_pos = viewport.toWorld(from.worldTransform.tx, from.worldTransform.ty);
            to = node.in_c[i].conn_line.to;
            to_pos = viewport.toWorld(to.worldTransform.tx, to.worldTransform.ty);
            update_line(node.in_c[i].conn_line, to_pos, from_pos) // Is inverted
            if (viewport.children.length > 0){
                viewport.setChildIndex(node.in_c[i].conn_line, viewport.children.length-1);
            }
        }
    }

    for (i=0; i<node.out_c.length; i++){
        for (j=0; j<node.out_c[i].conn_line.length; j++){
            if (node.out_c[i].conn_line[j]){ 
                from = node.out_c[i].conn_line[j].from;
                from_pos = viewport.toWorld(from.worldTransform.tx, from.worldTransform.ty);
                to = node.out_c[i].conn_line[j].to;
                to_pos = viewport.toWorld(to.worldTransform.tx, to.worldTransform.ty);
                update_line(node.out_c[i].conn_line[j], to_pos, from_pos) // Is inverted
                if (viewport.children.length > 0){
                    viewport.setChildIndex(node.out_c[i].conn_line[j], viewport.children.length-1);
                }
            }
        }
    }
}

function update_line(line, from, to){
    from_world = viewport.toWorld(from);
    line.clear();
    line.moveTo(from.x, from.y);
    line.lineStyle(3, 0x46b882, 1);

    var xctrl = (1.5 * to.x + 0.5 * from.x) / 2;
    var delta = to.y - from.y;
    var yctrl;
    if (delta < 0){
        yctrl = (to.y + from.y) / 2 + Math.min(Math.abs(delta), 50);
    } else {
        yctrl = (to.y + from.y) / 2 - Math.min(Math.abs(delta), 50);
    }

    line.quadraticCurveTo(xctrl, yctrl, to.x, to.y);
}

function name_to_size(name){
    var w = Math.max(60, 25 + name.length * 8);
    var h = 40; //name.length * 20;
    return [w, h];
}

function draw_rect(width, height, color, scale){
    var obj = new PIXI.Graphics();
    obj.lineStyle(2, 0x000000, 1);
    obj.beginFill(color);
    obj.drawRect(0, 0, width * scale, height * scale);
    obj.endFill();
    return obj;
}

function draw_block(name){
    var [width, height] = name_to_size(name);
    var obj = draw_rect(width, height, BLOCK_COLOR, 1);
    var text = draw_text(name);
    text.anchor.set(0.5, 0.5);
    text.position.set(obj.width / 2, obj.height / 2);
    obj.addChild(text);
    return [obj, text];
}

function draw_text(text, scale=1){
    text = new PIXI.Text(text,
        {
            fontFamily: FONT,
            fontSize: FONT_SIZE * scale,
            fill: TEXT_COLOR,
            align: 'right'
        });
    return text;
}

function draw_text_input(default_value, scale=1){
    var obj = new PIXI.TextInput({
        input: {
            fontFamily: FONT,
            fontSize: FONT_SIZE * scale,
            padding: '12px',
            width: '200px',
            height: '30px',
            color: INPUT_TEXT_COLOR,
        },
        box: {
            default: {fill: 0xE8E9F3, rounded: 16, stroke: {color: 0xCBCEE0, width: 4}},
            focused: {fill: 0xE1E3EE, rounded: 16, stroke: {color: 0xABAFC6, width: 4}},
            disabled: {fill: 0xDBDBDB, rounded: 16}
        }
    });
    obj.placeholder = default_value;
    return obj;
}

function draw_conn(inputs, outputs, rect){
    var width = rect.width;
    var height = rect.height;

    var in_even = ((inputs % 2 === 0) ? 1 : 0);
    var out_even = ((outputs % 2 === 0) ? 1 : 0);
    var in_step = width / inputs / (in_even + 1);
    var out_step = width / outputs / (out_even + 1);

    var radius = Math.max(6, 10 - (2 * inputs));

    var input_conn = [];
    var output_conn = [];

    function draw_circle(){
        var obj = new PIXI.Graphics();
        obj.lineStyle(2, 0x000000, 1);
        obj.beginFill(INPUT_COLOR);
        obj.drawCircle(0, 0, radius);
        obj.endFill();
        return obj;
    }

    var x = rect.position.x + width / 2 - 1;
    var y = height - 45;
    var offset = 0;
    for (var i = 0; i < inputs; i++){
        var obj = draw_circle()
        if (i !== 0 || inputs % 2 === 0) {
            offset = (-1)**i * (1 + Math.floor((i - 1 + in_even) / 2)) * in_step;
        }
        obj.position.set(x + offset, y);
        input_conn.push(obj);
    }

    y = height + 1;
    offset = 0;
    for (i = 0; i < outputs; i++){
        obj = draw_circle()
        if (i !== 0 || outputs % 2 === 0) {
            offset = (-1)**i * (1 + Math.floor((i - 1 + out_even) / 2)) * out_step;
        }
        obj.position.set(x + offset, y);
        output_conn.push(obj);
    }

    function sort_logic(x, y){
        if (x.position.x < y.position.x){
            return -1;
        } else if (x.position.x > y.position.x){
            return 1;
        }
        return 0;
    }

    input_conn.sort(sort_logic);
    output_conn.sort(sort_logic);
    return [input_conn, output_conn];
}
