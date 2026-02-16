const { app, BrowserWindow } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

let backendProcess = null;

function startBackend() {
  if (process.env.PYECN_BACKEND_DISABLED === "1") {
    return;
  }
  const python = process.env.PYECN_PYTHON || "python";
  const args = ["-m", "pyecn.browser_visualizations.live_api"];
  const cwd = path.resolve(__dirname, "..");
  const env = { ...process.env, PYTHONIOENCODING: "utf-8" };

  backendProcess = spawn(python, args, { cwd, env });

  backendProcess.stdout?.on("data", (data) => {
    process.stdout.write(String(data));
  });
  backendProcess.stderr?.on("data", (data) => {
    process.stderr.write(String(data));
  });
  backendProcess.on("exit", (code) => {
    console.log(`[pyecn] backend exited with code ${code}`);
  });
}

function stopBackend() {
  if (!backendProcess) return;
  backendProcess.kill();
  backendProcess = null;
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    backgroundColor: "#0b0f1a",
    webPreferences: {
      preload: path.join(__dirname, "preload.js")
    }
  });

  const devUrl = process.env.ELECTRON_DEV_URL;
  if (devUrl) {
    win.loadURL(devUrl);
  } else {
    const indexPath = path.join(__dirname, "..", "web", "dist", "index.html");
    win.loadFile(indexPath);
  }
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  stopBackend();
});
