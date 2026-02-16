# PyECN Live Electron

This folder wraps the React UI and starts the FastAPI backend automatically.

## Dev

```bash
cd electron
npm install
npm run dev
```

This starts the Vite dev server, launches Electron, and spawns the backend.

## Build

```bash
cd electron
npm run build
```

## Notes
- Set `PYECN_PYTHON` to point to the Python executable you want Electron to use.
- Set `PYECN_BACKEND_DISABLED=1` if you want to run the backend manually.
