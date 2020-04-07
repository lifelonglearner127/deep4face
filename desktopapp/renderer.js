// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// No Node.js APIs are available in this process because
// `nodeIntegration` is turned off. Use `preload.js` to
// selectively enable features needed in the rendering
// process.

var socket = io("http://127.0.0.1:6789/video");
socket.on('frame_download', (msg) => frame_queue.push(msg));

window.onload = function () {
    var frame_ctx = document.getElementById("canvas").getContext("2d");
    frame_ctx.font = "40px Black";
    frame_ctx.lineWidth = "2";
    frame_ctx.strokeStyle = "Yellow";

    [0, 1, 2, 3, 4].forEach((value, index) =>
        document.getElementById('camera_list_top').innerHTML += "<button onclick='sendIP(this)' name='Camera_Button' id='bt1' class='h-100 w-25 btn btn-outline-info' value='" + value + "'>Camera " + value + "</button>");

    [5, 6, 7, 8, 9].forEach((value, index) =>
        document.getElementById('camera_list_down').innerHTML += "<button onclick='sendIP(this)' name='Camera_Button' id='bt1' class='h-100 w-25 btn btn-outline-info' value='" + value + "'>Camera " + value + "</button>");

    setInterval(function () {
        let msg = frame_queue.shift();
        if (msg) {
            image.onload = drawFrame(msg, frame_ctx);
            image.src = getUrl(msg.frame);
        };
    }, 40);
}

var threshold = 0.97, img_width = 1121, img_height = 672, input_size = 672;
var frame_queue = [], prefix = 'data:image/jpeg;base64,';
var image = new Image()

var getUrl = (base_string) => { return prefix + base_string };

var drawFrame = (msg, context) => {
    context.drawImage(image, 0, 0, img_width, img_height);
}
