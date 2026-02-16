# PyECN Live Web

This is the React + Vite frontend for the PyECN live visualization API.

## Run locally

1. Start the API server (from the repo root):

```bash
python -m pyecn.browser_visualizations.live_api
```

2. Start the Vite dev server (from the repo root):

```bash
cd web
npm install
npm run dev
```

The frontend expects the API at http://127.0.0.1:8000 and proxies /api during development.
