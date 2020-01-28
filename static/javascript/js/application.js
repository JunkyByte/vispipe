let app = new PIXI.Application({
    antialias: true,
    autoResize: true,
    resolution: window.devicePixelRatio
});

app.renderer.backgroundColor = 0xc4c2be;
app.renderer.view.style.position = 'absolute';
app.renderer.view.style.display = 'block';
document.body.appendChild(app.view);

const rect = new PIXI.Graphics()
	.beginFill(0xff0000)
  .drawRect(-100, -100, 100, 100);
app.stage.addChild(rect);

// Listen for window resize events
window.addEventListener('resize', resize);

// Resize function window
function resize() {
    app.renderer.resize(window.innerWidth, window.innerHeight);
    rect.position.set(app.screen.width, app.screen.height);
}
resize();

$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');

    //receive details from server
    socket.on('newnumber', function(msg) {
        console.log("Received number" + msg.number);
        //$('#log').html(numbers_string);
        socket.emit('test_receive', 'test_send_see_me_python')
    });

});
