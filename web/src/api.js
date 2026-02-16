export async function runSimulation(configFile, currentFile) {
  const form = new FormData();
  form.append("config", configFile);
  if (currentFile) {
    form.append("current", currentFile);
  }
  const res = await fetch("/api/sim/run", { method: "POST", body: form });
  return res.json();
}

export async function loadResults(resultsFile) {
  const form = new FormData();
  form.append("results", resultsFile);
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

export async function fetchFrame(timeIndex) {
  const res = await fetch(`/api/plots/frame?time_index=${timeIndex}`);
  return res.json();
}
