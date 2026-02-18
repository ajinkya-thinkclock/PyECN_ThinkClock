export async function runSimulation(configFile, currentFile, measuredFile, rctPercent) {
  const form = new FormData();
  form.append("config", configFile);
  if (currentFile) {
    form.append("current", currentFile);
  }
  if (measuredFile) {
    form.append("measured", measuredFile);
  }
  if (rctPercent !== undefined && rctPercent !== null && rctPercent !== "") {
    form.append("rct_percent", rctPercent);
  }
  const res = await fetch("/api/sim/run", { method: "POST", body: form });
  return res.json();
}

export async function loadResults(resultsFile, measuredFile) {
  const form = new FormData();
  form.append("results", resultsFile);
  if (measuredFile) {
    form.append("measured", measuredFile);
  }
  const res = await fetch("/api/results/load", { method: "POST", body: form });
  return res.json();
}

export async function fetchStatus() {
  const res = await fetch("/api/sim/status");
  return res.json();
}

export async function fetchLog() {
  const res = await fetch("/api/sim/log");
  return res.json();
}

export async function fetchFrame(timeIndex, rctIndex = 0) {
  const res = await fetch(`/api/plots/frame?time_index=${timeIndex}&rct_index=${rctIndex}`);
  return res.json();
}
