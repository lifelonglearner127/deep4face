// Modules to control application life and create native browser window
const {app, BrowserWindow} = require('electron')
const path = require('path')

function createWindow () {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 730,
    frame: false,
    resizable: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      zoomFactor: 0.66
    }
  })

  // and load the index.html of the app.
  mainWindow.loadFile('index.html')

  // Open the DevTools.
  // mainWindow.webContents.openDevTools()
}
app.allowRendererProcessReuse = true
// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', function () {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
const server = require('http').createServer(app)
const io = require('socket.io')(server)
var lunatic_current_ip = '0';

io.of('/video').on('connection', (socket) => {
  console.log('a lunatic client connected');

  socket.on('frame_data', (msg) => {
    socket.emit('response', lunatic_current_ip)
    if (msg.frame != 0) {
      socket.broadcast.emit('frame_download', {
        'frame': msg.frame.toString('base64')
      });
    }
  });
  socket.on('disconnect', () => { console.log('a lunatic client disconnected') });
});

server.listen(6789, () => console.log('listening on *:6789'));
