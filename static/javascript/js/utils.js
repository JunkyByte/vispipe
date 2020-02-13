function onDragStart(event)
{
    // store a reference to the data
    // the reason for this is because of multitouch
    // we want to track the movement of this particular touch
    var i;
    var child;

    this.target = event.target;
    this.ischild = false;
    for (i=0; i<event.target.children.length; i++){
        this.child = event.target.children[i];
        if (this.child.text === undefined && this.child.containsPoint(event.data.global)) { // TODO: Find better way than checking text
            var obj = new PIXI.Graphics();
            this.ischild = obj;
            app.stage.addChildAt(this.ischild, app.stage.children.length);
            break;
        }
    }

    this.start_pos = new PIXI.Point(event.data.global.x, event.data.global.y);
    this.data = event.data;
    this.dragging = this.data.getLocalPosition(this.parent);

    if (!this.ischild){
        this.alpha = 0.5;
        app.stage.setChildIndex(this, app.stage.children.length-1);
    } else {
        app.stage.setChildIndex(this, app.stage.children.length-2);
    }
}

function onDragEnd(event)
{
    // Close drag logic
    this.alpha = 1;
    this.dragging = false;
    // set the interaction data to null
    this.data = null;

    if (this.ischild){
        // START FROM HERE -> in this.target you have original obj. run a search (on point) on each dynamic node child.
        // this.target.node is a reference to the node (with id)
        console.log(this.child.index, this.child.type);
        var target_node = point_to_node(event.data.global);
        var target_parent = target_node.parent.node;
        if (target_node && target_node.type !== this.child.type){
            console.log('Compatible connection');
            var input = (this.child.type === 'input') ? this.child : target_node;
            var output = (this.child.type === 'output') ? this.child : target_node;
            // There are two things to consider:
            // If we connect a input node which is already connected we need to remap
            // its connection to the new output
            // If we connect a output node which is already connected we need to APPEND
            // its new connection
            if (input.connection){
                console.log('Already connected input');
                // TODO: MANAGE HERE the remap
            }
            input.connection = output;
            output.connection.push(input);
            this.ischild.from = input;
            this.ischild.to = output;
            input.line = this.ischild;
            output.line.push(this.ischild);
            // TODO: ADD ACTUAL CONN LOGIC (PIPELINE CLASS)
        } else {
            this.ischild.destroy();
        }
    }

    this.ischild = false;
    this.child = null;
    this.target = null;
}

function point_to_node(point){
    var i, child;
    var root_obj = app.renderer.plugins.interaction.hitTest(point);

    for (i=0; i<root_obj.children.length; i++){
        child = root_obj.children[i];
        if (child.text === undefined && child.containsPoint(point)) { // TODO: Find bw than text
            return child;
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
        update_line(this.ischild, this.start_pos, event.data.global);
    }
}

function update_all_lines(node){
    var i, j, from, to, from_pos, to_pos;

    for (i=0; i<node.in_c.length; i++){
        if (node.in_c[i].line){
            from = node.in_c[i].line.from;
            from_pos = new PIXI.Point(from.worldTransform.tx, from.worldTransform.ty);
            to = node.in_c[i].line.to;
            to_pos = new PIXI.Point(to.worldTransform.tx, to.worldTransform.ty);
            update_line(node.in_c[i].line, from_pos, to_pos)
            app.stage.setChildIndex(node.in_c[i].line, app.stage.children.length-1);
        }
    }

    for (i=0; i<node.out_c.length; i++){
        for (j=0; j<node.out_c[i].line.length; j++){
            if (node.out_c[i].line[j]){ 
                from = node.out_c[i].line[j].from;
                from_pos = new PIXI.Point(from.worldTransform.tx, from.worldTransform.ty);
                to = node.out_c[i].line[j].to;
                to_pos = new PIXI.Point(to.worldTransform.tx, to.worldTransform.ty);
                update_line(node.out_c[i].line[j], from_pos, to_pos)
                app.stage.setChildIndex(node.out_c[i].line[j], app.stage.children.length-1);
            }
        }
    }
}

function update_line(line, from, to){
    line.clear();
    line.moveTo(from.x, from.y);
    line.lineStyle(3, 0x46b882, 1);

    var xctrl = (1.5 * to.x + 0.5 * from.x) / 2;
    var delta = to.y - from.y;
    if (delta < 0){
        var yctrl = (to.y + from.y) / 2 + Math.min(Math.abs(delta), 50);
    } else {
        var yctrl = (to.y + from.y) / 2 - Math.min(Math.abs(delta), 50);
    }

    line.quadraticCurveTo(xctrl, yctrl, to.x, to.y);
}

function name_to_size(name){
    var w = Math.max(60, 25 + name.length * 8);
    var h = 40; //name.length * 20;
    return [w, h];
}

function draw_rect(width, height, color, scale=1){
    var obj = new PIXI.Graphics();
    obj.lineStyle(2, 0x000000, 1);
    obj.beginFill(color);
    obj.drawRect(0, 0, width * scale, height * scale);
    obj.endFill();
    return obj
}

function draw_block(name){
    var [width, height] = name_to_size(name);
    var obj = draw_rect(width, height, BLOCK_COLOR, 1);
    var text = draw_text(name);
    text.anchor.set(0.5, 0.5);
    text.position.set(obj.width / 2, obj.height / 2);
    obj.addChild(text);
    return [obj, text]
}

function draw_text(text, scale=1){
    text = new PIXI.Text(text,
        {
            fontFamily : FONT,
            fontSize: FONT_SIZE * scale,
            fill : TEXT_COLOR,
            align : 'right'
        });
    return text
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
        return obj
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

    return [input_conn, output_conn]
}
