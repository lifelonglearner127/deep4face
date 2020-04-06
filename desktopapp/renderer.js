// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// No Node.js APIs are available in this process because
// `nodeIntegration` is turned off. Use `preload.js` to
// selectively enable features needed in the rendering
// process.

window.onload = function () {

    [0, 1, 2, 3, 4].forEach((value, index) =>
        document.getElementById('camera_list_top').innerHTML += "<button onclick='sendIP(this)' name='Camera_Button' id='bt1' class='h-100 w-25 btn btn-outline-info' value='" + value + "'>Camera " + value + "</button>");

    [5, 6, 7, 8, 9].forEach((value, index) =>
        document.getElementById('camera_list_down').innerHTML += "<button onclick='sendIP(this)' name='Camera_Button' id='bt1' class='h-100 w-25 btn btn-outline-info' value='" + value + "'>Camera " + value + "</button>");
}