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
const HEATMAP_COLORSCALE = "RdBu";

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

function buildHeatmapPlot(map, title, colorbarTitle, xTitle, yTitle) {
  return {
    data: [
      {
        type: "heatmap",
        z: map.z,
        x: map.x,
        y: map.y,
        colorscale: HEATMAP_COLORSCALE,
        colorbar: { title: colorbarTitle }
      }
    ],
    layout: {
      ...makeLayout(title),
      xaxis: { title: xTitle },
      yaxis: { title: yTitle }
    }
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
  const [measuredFile, setMeasuredFile] = useState(null);
  const [resultsFile, setResultsFile] = useState(null);
  const [actionMessage, setActionMessage] = useState("Idle");
  const [rctPercent, setRctPercent] = useState("");
  const [rctIndex, setRctIndex] = useState(0);

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
      const data = await fetchFrame(timeIndex, rctIndex);
      setFrame(data);
    };
    load();
  }, [timeIndex, nt, rctIndex]);

  const tempMapElbPlot = useMemo(() => {
    const temp = frame?.temp_maps?.elb || frame?.temp_maps?.combined;
    if (!temp) return null;
    return buildHeatmapPlot(temp, "Electrode Temperature (Elb)", "C", "Unrolled/Width", "Axial/Height");
  }, [frame]);

  const tempMapElrPlot = useMemo(() => {
    const temp = frame?.temp_maps?.elr;
    if (!temp) return null;
    return buildHeatmapPlot(temp, "Electrode Temperature (Elr)", "C", "Unrolled/Width", "Axial/Height");
  }, [frame]);

  const socHeatmapElbPlot = useMemo(() => {
    const soc = frame?.soc_heatmaps?.elb || frame?.soc_heatmaps?.combined;
    if (!soc) return null;
    const traces = [
      {
        type: "heatmap",
        z: soc.z,
        x: soc.x,
        y: soc.y,
        colorscale: HEATMAP_COLORSCALE,
        colorbar: { title: "SoC %" }
      }
    ];
    if (soc.grad) {
      traces.push({
        type: "heatmap",
        z: soc.grad,
        x: soc.x,
        y: soc.y,
        colorscale: HEATMAP_COLORSCALE,
        opacity: 0.35,
        showscale: false,
        hoverinfo: "skip"
      });
    }
    return {
      data: traces,
      layout: { ...makeLayout("SoC Heatmap (Elb)"), xaxis: { title: "Unrolled Distance" }, yaxis: { title: "Axial Position" } }
    };
  }, [frame]);

  const socHeatmapElrPlot = useMemo(() => {
    const soc = frame?.soc_heatmaps?.elr;
    if (!soc) return null;
    const traces = [
      {
        type: "heatmap",
        z: soc.z,
        x: soc.x,
        y: soc.y,
        colorscale: HEATMAP_COLORSCALE,
        colorbar: { title: "SoC %" }
      }
    ];
    if (soc.grad) {
      traces.push({
        type: "heatmap",
        z: soc.grad,
        x: soc.x,
        y: soc.y,
        colorscale: HEATMAP_COLORSCALE,
        opacity: 0.35,
        showscale: false,
        hoverinfo: "skip"
      });
    }
    return {
      data: traces,
      layout: { ...makeLayout("SoC Heatmap (Elr)"), xaxis: { title: "Unrolled Distance" }, yaxis: { title: "Axial Position" } }
    };
  }, [frame]);

  const voltagePlot = useMemo(() => {
    const v = frame?.voltage;
    if (!v) return null;
    const measured = frame?.voltage_measured;
    const traces = [{ type: "scatter", mode: "lines", x: v.time, y: v.values, name: "Simulated" }];
    if (measured && measured.time && measured.values) {
        traces.push({
          type: "scatter",
          mode: "lines",
          x: measured.time,
          y: measured.values,
          name: "Measured",
          line: { color: "#f97316", width: 2 }
        });
    }
    return {
      data: traces,
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

  const currentDensityElbPlot = useMemo(() => {
    const cd = frame?.current_density_maps?.elb || frame?.current_density_maps?.combined;
    if (!cd) return null;
    return buildHeatmapPlot(cd, "Current Density (Elb)", "A/m2", "Unrolled Distance", "Axial Position");
  }, [frame]);

  const currentDensityElrPlot = useMemo(() => {
    const cd = frame?.current_density_maps?.elr;
    if (!cd) return null;
    return buildHeatmapPlot(cd, "Current Density (Elr)", "A/m2", "Unrolled Distance", "Axial Position");
  }, [frame]);

  const rctList = useMemo(() => {
    const rct = frame?.rct_series;
    if (!rct) return null;
    if (!Array.isArray(rct.values) || rct.values.length === 0) {
      return null;
    }
    return rct.values.map((value, idx) => `Node ${idx}: ${value}`);
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
    const res = await runSimulation(configFile, currentFile, measuredFile, rctPercent);
    setActionMessage(res.message || res.status || "Run started.");
  };

  const handleLoad = async () => {
    if (!resultsFile) {
      setActionMessage("Please select a results NPZ file.");
      return;
    }
    const res = await loadResults(resultsFile, measuredFile);
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
          <label className="file-input">
            <span>Measured Voltage CSV (optional)</span>
            <input type="file" accept=".csv" onChange={(e) => setMeasuredFile(e.target.files?.[0] || null)} />
          </label>
          <label className="file-input">
            <span>Rct spread (%)</span>
            <input
              type="number"
              min="0"
              step="0.1"
              placeholder="0"
              value={rctPercent}
              onChange={(e) => setRctPercent(e.target.value)}
            />
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
          <div className="diag-card">
            <div className="diag-title">Rct values (selected RC)</div>
            <pre>{rctList ? rctList.join("\n") : "No Rct scale data"}</pre>
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
        <div className="playback">
          <div className="speed">
            <span>Rct branch</span>
            <select value={rctIndex} onChange={(e) => setRctIndex(Number(e.target.value))}>
              {Array.from({ length: frame?.rct_series?.rc_count || 1 }, (_, idx) => (
                <option key={idx} value={idx}>
                  RC {idx + 1}
                </option>
              ))}
            </select>
          </div>
        </div>
      </section>

      <section className="plots">
        <PlotCard title="Electrode Temperature (Elb)">
          {tempMapElbPlot ? (
            <Plot data={tempMapElbPlot.data} layout={tempMapElbPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="Electrode Temperature (Elr)">
          {tempMapElrPlot ? (
            <Plot data={tempMapElrPlot.data} layout={tempMapElrPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="Current Density (Elb)">
          {currentDensityElbPlot ? (
            <Plot data={currentDensityElbPlot.data} layout={currentDensityElbPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="Current Density (Elr)">
          {currentDensityElrPlot ? (
            <Plot data={currentDensityElrPlot.data} layout={currentDensityElrPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="SoC Heatmap (Elb)">
          {socHeatmapElbPlot ? (
            <Plot data={socHeatmapElbPlot.data} layout={socHeatmapElbPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
          ) : (
            <EmptyPlot />
          )}
        </PlotCard>
        <PlotCard title="SoC Heatmap (Elr)">
          {socHeatmapElrPlot ? (
            <Plot data={socHeatmapElrPlot.data} layout={socHeatmapElrPlot.layout} className="plot" useResizeHandler style={{ width: "100%", height: "100%" }} />
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
