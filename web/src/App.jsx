import React, { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import {
  fetchFrame,
  fetchLog,
  fetchStatus,
  loadResults,
  runSimulation
} from "./api";

const SPEED_OPTIONS = [1, 2, 5, 10, 20];
const PLAYBACK_INTERVAL_MS = 200;
const STEP_MULTIPLIER = 5;

function EmptyPlot({ note }) {
  return (
    <div className="plot-empty">
      <div className="plot-empty-note">{note || "Waiting for data"}</div>
    </div>
  );
}

function PlotCard({ title, children }) {
  return (
    <div className="plot-card">
      <div className="plot-header">{title}</div>
      <div className="plot-body">{children}</div>
    </div>
  );
}

function makeLayout(title) {
  return {
    title,
    margin: { l: 48, r: 20, t: 48, b: 40 },
    paper_bgcolor: "#10151f",
    plot_bgcolor: "#10151f",
    font: { color: "#e5e7eb" },
    uirevision: "keep"
  };
}

export default function App() {
  const [status, setStatus] = useState({});
  const [logText, setLogText] = useState("");
  const [meta, setMeta] = useState({});
  const [timeIndex, setTimeIndex] = useState(0);
  const [nt, setNt] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [frame, setFrame] = useState(null);
  const [configFile, setConfigFile] = useState(null);
  const [currentFile, setCurrentFile] = useState(null);
  const [resultsFile, setResultsFile] = useState(null);
  const [actionMessage, setActionMessage] = useState("Idle");

  useEffect(() => {
    const handle = setInterval(async () => {
      const s = await fetchStatus();
      setStatus(s);
      setMeta(s.results_meta || {});
      setNt(s.nt || 0);
      if (s.nt) {
        setTimeIndex((prev) => Math.min(prev, Math.max(0, s.nt - 1)));
      }
      const log = await fetchLog();
      setLogText(log.log || "");
    }, 1000);
    return () => clearInterval(handle);
  }, []);

  useEffect(() => {
    if (!playing) return undefined;
    const step = Math.max(1, speed * STEP_MULTIPLIER);
    const handle = setInterval(() => {
      setTimeIndex((prev) => {
        if (!nt) return prev;
        const next = prev + step;
        return next >= nt ? nt - 1 : next;
      });
    }, PLAYBACK_INTERVAL_MS);
    return () => clearInterval(handle);
  }, [playing, speed, nt]);

  useEffect(() => {
    const load = async () => {
      if (!nt) {
        setFrame(null);
        return;
      }
      const data = await fetchFrame(timeIndex);
      setFrame(data);
    };
    load();
  }, [timeIndex, nt]);

  const tempMapPlot = useMemo(() => {
    const temp = frame?.temp_map;
    if (!temp) return null;
    return {
      data: [
        {
          type: "heatmap",
          z: temp.z,
          x: temp.x,
          y: temp.y,
          colorscale: "RdBu",
          zsmooth: "best",
          colorbar: { title: "C" }
        }
      ],
      layout: { ...makeLayout("Electrode Temperature Map"), xaxis: { title: "Unrolled/Width" }, yaxis: { title: "Axial/Height" } }
    };
  }, [frame]);

  const socHeatmapPlot = useMemo(() => {
    const soc = frame?.soc_heatmap;
    if (!soc) return null;
    const traces = [
      {
        type: "heatmap",
        z: soc.z,
        x: soc.x,
        y: soc.y,
        colorscale: "Viridis",
        zsmooth: "best",
        colorbar: { title: "SoC %" }
      }
    ];
    if (soc.grad) {
      traces.push({
        type: "heatmap",
        z: soc.grad,
        x: soc.x,
        y: soc.y,
        colorscale: "Turbo",
        zsmooth: "best",
        opacity: 0.35,
        showscale: false,
        hoverinfo: "skip"
      });
    }
    return {
      data: traces,
      layout: { ...makeLayout("SoC Heatmap"), xaxis: { title: "Unrolled Distance" }, yaxis: { title: "Axial Position" } }
    };
  }, [frame]);

  const voltagePlot = useMemo(() => {
    const v = frame?.voltage;
    if (!v) return null;
    return {
      data: [{ type: "scatter", mode: "lines", x: v.time, y: v.values, name: "Voltage" }],
      layout: { ...makeLayout("Voltage vs Time"), xaxis: { title: "Time (s)" }, yaxis: { title: "Voltage (V)" } }
    };
  }, [frame]);

  const currentPlot = useMemo(() => {
    const c = frame?.current;
    if (!c) return null;
    const values = Array.isArray(c.values) ? [...c.values] : c.values;
    if (Array.isArray(values) && values.length > 1) {
      values[0] = values[1];
    }
    return {
      data: [{ type: "scatter", mode: "lines", x: c.time, y: values, name: "Current" }],
      layout: { ...makeLayout("Current vs Time"), xaxis: { title: "Time (s)" }, yaxis: { title: "Current (A)" } }
    };
  }, [frame]);

  const socPlot = useMemo(() => {
    const s = frame?.soc;
    if (!s) return null;
    return {
      data: [{ type: "scatter", mode: "lines", x: s.time, y: s.values, name: "SoC" }],
      layout: { ...makeLayout("State of Charge"), xaxis: { title: "Time (s)" }, yaxis: { title: "SoC (%)" } }
    };
  }, [frame]);

  const currentDensityPlot = useMemo(() => {
    const cd = frame?.current_density;
    if (!cd) return null;
    return {
      data: [
        {
          type: "heatmap",
          z: cd.z,
          x: cd.x || undefined,
          y: cd.y || undefined,
          colorscale: "Blues",
          zsmooth: "best",
          colorbar: { title: "A/m2" }
        }
      ],
      layout: { ...makeLayout("Current Density"), xaxis: { title: "Unrolled Distance" }, yaxis: { title: "Axial Position" } }
    };
  }, [frame]);

  const heatgenPlot = useMemo(() => {
    const h = frame?.heatgen;
    if (!h) return null;
    return {
      data: [{ type: "scatter", mode: "lines", x: h.time, y: h.values, name: "Heat Gen" }],
      layout: { ...makeLayout("Heat Generation vs Time"), xaxis: { title: "Time (s)" }, yaxis: { title: "W" } }
    };
  }, [frame]);

  const tempStatsPlot = useMemo(() => {
    const t = frame?.temp_stats;
    if (!t) return null;
    return {
      data: [
        { type: "scatter", mode: "lines", x: t.time, y: t.avg, name: "Avg" },
        { type: "scatter", mode: "lines", x: t.time, y: t.min, name: "Min" },
        { type: "scatter", mode: "lines", x: t.time, y: t.max, name: "Max" }
      ],
      layout: { ...makeLayout("Temperature Min/Max/Avg"), xaxis: { title: "Time (s)" }, yaxis: { title: "C" } }
    };
  }, [frame]);

  const handleRun = async () => {
    if (!configFile) {
      setActionMessage("Please select a config TOML file.");
      return;
    }
    const res = await runSimulation(configFile, currentFile);
    setActionMessage(res.message || res.status || "Run started.");
  };

  const handleLoad = async () => {
    if (!resultsFile) {
      setActionMessage("Please select a results NPZ file.");
      return;
    }
    const res = await loadResults(resultsFile);
    setActionMessage(res.message || res.status || "Results loaded.");
  };

  return (
    <div className="app">
      <header className="hero">
        <div>
          <div className="eyebrow">PyECN Live</div>
          <h1>Battery simulation cockpit</h1>
          <p>Run PyECN jobs, stream results, and explore spatial maps in real time.</p>
        </div>
        <div className="status-card">
          <div className="status-title">Status</div>
          <div className="status-line">{status.last_message || "Idle"}</div>
          {status.last_error && <div className="status-error">{status.last_error}</div>}
          <div className="status-meta">nt: {nt || 0}</div>
        </div>
      </header>

      <section className="panel">
        <div className="panel-title">Run or Load</div>
        <div className="panel-grid">
          <label className="file-input">
            <span>Config TOML</span>
            <input type="file" accept=".toml" onChange={(e) => setConfigFile(e.target.files?.[0] || null)} />
          </label>
          <label className="file-input">
            <span>Current CSV (optional)</span>
            <input type="file" accept=".csv" onChange={(e) => setCurrentFile(e.target.files?.[0] || null)} />
          </label>
          <button className="primary" onClick={handleRun}>
            Run Simulation
          </button>
          <label className="file-input">
            <span>Results NPZ</span>
            <input type="file" accept=".npz" onChange={(e) => setResultsFile(e.target.files?.[0] || null)} />
          </label>
          <button className="ghost" onClick={handleLoad}>
            Load Results
          </button>
          <div className="action-message">{actionMessage}</div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-title">Diagnostics</div>
        <div className="diag-grid">
          <div className="diag-card">
            <div className="diag-title">Run log</div>
            <pre>{logText || "(no logs yet)"}</pre>
          </div>
          <div className="diag-card">
            <div className="diag-title">Results meta</div>
            <pre>{Object.keys(meta || {}).length ? JSON.stringify(meta, null, 2) : "(none)"}</pre>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-title">Playback</div>
        <div className="playback">
          <button className="primary" onClick={() => setPlaying(true)}>
            Play
          </button>
          <button className="ghost" onClick={() => setPlaying(false)}>
            Pause
          </button>
          <input
            type="range"
            min="0"
            max={Math.max(0, nt - 1)}
            value={timeIndex}
            onChange={(e) => setTimeIndex(Number(e.target.value))}
          />
          <div className="speed">
            <span>Speed</span>
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
              {SPEED_OPTIONS.map((val) => (
                <option key={val} value={val}>
                  {val}x
                </option>
              ))}
            </select>
          </div>
          <div className="frame-note">Frame {timeIndex} / {Math.max(0, nt - 1)}</div>
        </div>
      </section>

      <section className="plots">
        <PlotCard title="Electrode Temperature Map">
          {tempMapPlot ? (
            <Plot data={tempMapPlot.data} layout={tempMapPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="Current Density">
          {currentDensityPlot ? (
            <Plot data={currentDensityPlot.data} layout={currentDensityPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="SoC Heatmap">
          {socHeatmapPlot ? (
            <Plot data={socHeatmapPlot.data} layout={socHeatmapPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="Voltage vs Time">
          {voltagePlot ? (
            <Plot data={voltagePlot.data} layout={voltagePlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="Current vs Time">
          {currentPlot ? (
            <Plot data={currentPlot.data} layout={currentPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="State of Charge">
          {socPlot ? (
            <Plot data={socPlot.data} layout={socPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="Heat Generation vs Time">
          {heatgenPlot ? (
            <Plot data={heatgenPlot.data} layout={heatgenPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="Temperature Min/Max/Avg">
          {tempStatsPlot ? (
            <Plot data={tempStatsPlot.data} layout={tempStatsPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
      </section>
    </div>
  );
}
