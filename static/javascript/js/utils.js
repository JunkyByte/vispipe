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
    w = name.length * 35;
    h = 75; //name.length * 20;
    return [w, h];
}

function draw_block(name){
    var [width, height] = name_to_size(name);
    var obj = new PIXI.Graphics();
    obj.lineStyle(2, 0x000000, 1);
    obj.beginFill(0x9eef92);
    obj.drawRect(0, 0, width, height);
    obj.endFill();
    return obj
}

