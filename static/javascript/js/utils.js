function onDragStart(event)
{
    // store a reference to the data
    // the reason for this is because of multitouch
    // we want to track the movement of this particular touch
    this.data = event.data;
    this.alpha = 0.5;
    this.dragging = this.data.getLocalPosition(this.parent);
}

function onDragEnd()
{
    this.alpha = 1;
    this.dragging = false;
    // set the interaction data to null
    this.data = null;
}

function onDragMove()
{
    if (this.dragging)
    {
        var newPosition = this.data.getLocalPosition(this.parent);
        this.position.x += (newPosition.x - this.dragging.x);
        this.position.y += (newPosition.y - this.dragging.y);
        this.dragging = newPosition;
    }
}

function name_to_size(name){
    w = 35 + name.length * 8;
    h = 40; //name.length * 20;
    return [w, h]
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
    var obj = draw_rect(width, height, color=BLOCK_COLOR, scale=1);
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