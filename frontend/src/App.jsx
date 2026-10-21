
import React, {
  useState,
  useEffect,
  useCallback,
  useMemo
} from "react";

import {
  Viewer,
  Entity as CesiumEntity,
  useCesium
} from "resium";

import {
  Cartesian2,
  Cartesian3,
  Color,
  Ion,
  ImageryLayer,
  Terrain,
  VerticalOrigin
} from "cesium";

import Plot from "react-plotly.js";

import {
  Satellite,
  Globe2,
  Radio,
  Activity,
  ShieldAlert,
  Brain,
  History,
  Settings,
  Play,
  Square,
  RefreshCw
} from "lucide-react";

// Cesium ion token from frontend/.env
Ion.defaultAccessToken =
  import.meta.env.VITE_CESIUM_ION_TOKEN || "";

const API = "/api";

const MENU = [
  ["Overview", Activity],
  ["Satellite Explorer", Satellite],
  ["Ground Stations", Globe2],
  ["Link Monitoring", Radio],
  ["Failure Simulator", ShieldAlert],
  ["Jamming Detection", Brain],
  ["Link Prediction", Activity],
  ["Experiment History", History],
  ["About and Data Sources", Settings]
];

const SCENARIOS = [
  "normal",
  "degradation",
  "high_latency",
  "packet_loss",
  "interference",
  "outage",
  "recovery",
  "handover"
];

const DEFAULT_THRESHOLDS = {
  latency: 180,
  packet_loss: 5,
  jitter: 30,
  snr: 10
};

const chartLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#cbd5e1" },
  margin: { l: 45, r: 15, t: 20, b: 40 },
  xaxis: { gridcolor: "#263449" },
  yaxis: { gridcolor: "#263449" },
  autosize: true
};

// -------------------------------
// Backend API helpers
// -------------------------------

async function request(path, options = {}) {
  const response = await fetch(API + path, {
    headers: {
      "Content-Type": "application/json"
    },
    ...options
  });

  const raw = await response.text();

  let data;

  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(
      `API returned an invalid response (${response.status}). ` +
      raw.slice(0, 150)
    );
  }

  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || data)
    );
  }

  return data;
}

function post(path, data = {}) {
  return request(path, {
    method: "POST",
    body: JSON.stringify(data)
  });
}

// -------------------------------
// Reusable dashboard components
// -------------------------------

function Card({ title, children }) {
  return (
    <section className="card">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Metric({ label, value, unit = "" }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>
        {value ?? "--"} {unit}
      </strong>
    </div>
  );
}

function Chart({
  title,
  history,
  field,
  color = "#38bdf8"
}) {
  return (
    <Card title={title}>
      <Plot
        data={[
          {
            x: history.map((x, i) => i + 1),
            y: history.map(x => x[field]),
            type: "scatter",
            mode: "lines+markers",
            line: {
              color,
              width: 2
            },
            marker: {
              size: 4
            }
          }
        ]}
        layout={{
          ...chartLayout,
          height: 250
        }}
        config={{
          responsive: true,
          displaylogo: false
        }}
        useResizeHandler
        style={{ width: "100%" }}
      />
    </Card>
  );
}

// -------------------------------
// Realistic Cesium Earth settings
// -------------------------------

function EarthSettings() {
  const { viewer } = useCesium();

  useEffect(() => {
    if (!viewer || viewer.isDestroyed()) return;

    const scene = viewer.scene;

   
scene.globe.show = true;
scene.globe.baseColor = Color.DARKBLUE;
scene.globe.enableLighting = false;

viewer.camera.flyHome(0);
scene.requestRender();


    // Earth atmosphere
    scene.globe.showGroundAtmosphere = true;

    if (scene.skyAtmosphere) {
      scene.skyAtmosphere.show = true;
    }

    // Sun and Moon
    if (scene.sun) {
      scene.sun.show = true;
    }

    if (scene.moon) {
      scene.moon.show = true;
    }

    // Start with a full-Earth camera view
    viewer.camera.setView({
      destination: Cartesian3.fromDegrees(
        77.5946,
        12.9716,
        20000000
      )
    });

    scene.requestRender();
  }, [viewer]);

  return null;
}

// -------------------------------
// 3D Globe
// -------------------------------

function Globe({
  satellites,
  selected,
  onSelect,
  station
}) {
  const imagery = useMemo(
    () => ImageryLayer.fromWorldImagery(),
    []
  );

  const terrain = useMemo(
    () =>
      Terrain.fromWorldTerrain({
        requestVertexNormals: true,
        requestWaterMask: true
      }),
    []
  );

  return (
    <div className="globe">
      <Viewer
        full
        baseLayer={imagery}
        terrain={undefined}
        baseLayerPicker={false}
        animation={false}
        timeline={false}
        geocoder={false}
        homeButton={true}
        navigationHelpButton={false}
        sceneModePicker={false}
        shouldAnimate={true}
        infoBox={true}
        selectionIndicator={true}
      >
        <EarthSettings />

        {/* Satellite positions */}
        {satellites.map(sat => {
          const longitude = Number(sat.longitude);
          const latitude = Number(sat.latitude);
          const altitude =
            Number(sat.altitude_km) * 1000;

          if (
            !Number.isFinite(longitude) ||
            !Number.isFinite(latitude) ||
            !Number.isFinite(altitude)
          ) {
            return null;
          }

          const isSelected = sat.id === selected;

          return (
            <CesiumEntity
              key={sat.id}
              name={sat.name}
              description={
                `Satellite: ${sat.name}<br/>` +
                `NORAD ID: ${sat.id}<br/>` +
                `Altitude: ${sat.altitude_km} km`
              }
              position={Cartesian3.fromDegrees(
                longitude,
                latitude,
                altitude
              )}
              point={{
                pixelSize: isSelected ? 13 : 7,
                color: isSelected
                  ? Color.YELLOW
                  : Color.CYAN,
                outlineColor: Color.BLACK,
                outlineWidth: 2,
                disableDepthTestDistance:
                  Number.POSITIVE_INFINITY
              }}
              label={
                isSelected
                  ? {
                      text: sat.name,
                      font: "14px sans-serif",
                      fillColor: Color.WHITE,
                      outlineColor: Color.BLACK,
                      outlineWidth: 2,
                      pixelOffset: new Cartesian2(0, -22),
                      verticalOrigin: VerticalOrigin.BOTTOM
                    }
                  : undefined
              }
              onClick={() => onSelect(sat.id)}
            />
          );
        })}

        {/* Ground station */}
        <CesiumEntity
          name="Ground Station"
          description="Demonstration ground station"
          position={Cartesian3.fromDegrees(
            Number(station.longitude),
            Number(station.latitude),
            Number(station.altitude_m)
          )}
          point={{
            pixelSize: 12,
            color: Color.LIME,
            outlineColor: Color.WHITE,
            outlineWidth: 2,
            disableDepthTestDistance:
              Number.POSITIVE_INFINITY
          }}
          label={{
            text: "Ground Station",
            font: "14px sans-serif",
            fillColor: Color.WHITE,
            outlineColor: Color.BLACK,
            outlineWidth: 2,
            pixelOffset: new Cartesian2(0, -20),
            verticalOrigin: VerticalOrigin.BOTTOM,
            disableDepthTestDistance:
              Number.POSITIVE_INFINITY
          }}
        />
      </Viewer>
    </div>
  );
}

// -------------------------------
// Main application
// -------------------------------

export default function App() {
  const [page, setPage] = useState("Overview");

  const [satellites, setSatellites] = useState([]);
  const [source, setSource] = useState(null);
  const [selected, setSelected] = useState("");
  const [search, setSearch] = useState("");

  const [station, setStation] = useState({
    latitude: 12.9716,
    longitude: 77.5946,
    altitude_m: 920,
    elevation_mask: 10
  });

  const [visibility, setVisibility] = useState(null);

  const [running, setRunning] = useState(false);
  const [scenario, setScenario] = useState("normal");
  const [seed, setSeed] = useState(42);

  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [events, setEvents] = useState([]);

  const [thresholds, setThresholds] =
    useState(DEFAULT_THRESHOLDS);

  const [prediction, setPrediction] = useState(null);
  const [linkPrediction, setLinkPrediction] = useState(null);
  const [evaluation, setEvaluation] = useState(null);

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const chosen = satellites.find(
    s => s.id === selected
  );

  const filtered = satellites.filter(s =>
    s.name.toLowerCase().includes(
      search.toLowerCase()
    ) ||
    String(s.id).includes(search)
  );

  // -------------------------------
  // Load orbital data
  // -------------------------------

  const loadSatellites = useCallback(async () => {
    try {
      const result = await request("/satellites");

      setSatellites(result.satellites || []);
      setSource(result);

      setSelected(previous => {
        if (
          previous &&
          result.satellites.some(
            s => s.id === previous
          )
        ) {
          return previous;
        }

        return result.satellites[0]?.id || "";
      });

      if (result.fetch_error) {
        setError(
          "Orbital refresh failed; cached data displayed: " +
          result.fetch_error
        );
      }
    } catch (err) {
      setError(err.message);
    }
  }, []);

  const loadEvents = useCallback(async () => {
    try {
      setEvents(
        await request("/events?limit=100")
      );
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    loadSatellites();
    loadEvents();

    request("/thresholds")
      .then(setThresholds)
      .catch(() => {});

    request("/ml/evaluation")
      .then(setEvaluation)
      .catch(() => {});

    const timer = setInterval(
      loadSatellites,
      60000
    );

    return () => clearInterval(timer);
  }, [loadSatellites, loadEvents]);

  // -------------------------------
  // Live simulation monitoring
  // -------------------------------

  useEffect(() => {
    if (!running) return;

    let busy = false;

    const timer = setInterval(async () => {
      if (busy) return;

      busy = true;

      try {
        const result = await request(
          "/monitoring/metrics"
        );

        if (result.latest) {
          setMetrics(result.latest);

          setHistory(previous =>
            [...previous, result.latest].slice(-60)
          );
        }
      } catch (err) {
        setError(err.message);
      } finally {
        busy = false;
      }
    }, 2000);

    return () => clearInterval(timer);
  }, [running]);

  // -------------------------------
  // Ground station visibility
  // -------------------------------

  async function checkVisibility() {
    if (!selected) return;

    try {
      setLoading(true);

      const result = await post("/visibility", {
        satellite_id: selected,
        latitude: Number(station.latitude),
        longitude: Number(station.longitude),
        altitude_m: Number(station.altitude_m),
        elevation_mask: Number(station.elevation_mask)
      });

      setVisibility(result);
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // -------------------------------
  // Simulation controls
  // -------------------------------

  async function startSimulation() {
    try {
      await post("/simulation/start", {
        seed: Number(seed),
        satellite_id: selected || null
      });

      setScenario("normal");
      setHistory([]);
      setMetrics(null);
      setRunning(true);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function stopSimulation() {
    try {
      await post("/simulation/stop");

      setRunning(false);
      loadEvents();
    } catch (err) {
      setError(err.message);
    }
  }

  async function changeScenario(value) {
    try {
      await post("/simulation/scenario", {
        scenario: value
      });

      setScenario(value);
      loadEvents();
    } catch (err) {
      setError(err.message);
    }
  }

  // -------------------------------
  // Threshold settings
  // -------------------------------

  async function saveThresholds() {
    try {
      const result = await post("/thresholds", {
        latency: Number(thresholds.latency),
        packet_loss: Number(thresholds.packet_loss),
        jitter: Number(thresholds.jitter),
        snr: Number(thresholds.snr)
      });

      setThresholds(result);
      setError("");

      alert("Thresholds saved");
    } catch (err) {
      setError(err.message);
    }
  }

  // -------------------------------
  // Machine learning
  // -------------------------------

  async function runML() {
    if (!metrics) return;

    const features = {
      latency: metrics.latency,
      packet_loss: metrics.packet_loss,
      jitter: metrics.jitter,
      snr: metrics.snr
    };

    try {
      const [a, b] = await Promise.all([
        post("/ml/jamming/predict", features),
        post("/ml/link/predict", features)
      ]);

      setPrediction(a);
      setLinkPrediction(b);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  // -------------------------------
  // Satellite selection panel
  // -------------------------------

  const satellitePanel = (
    <Card title="Satellite Selection">
      <input
        placeholder="Search satellite name or NORAD ID"
        value={search}
        onChange={e => setSearch(e.target.value)}
      />

      <select
        value={selected}
        onChange={e => {
          setSelected(e.target.value);
          setVisibility(null);
        }}
      >
        {filtered.map(s => (
          <option key={s.id} value={s.id}>
            {s.name} ({s.id})
          </option>
        ))}
      </select>

      {chosen && (
        <div className="details">
          <Metric
            label="Satellite"
            value={chosen.name}
          />

          <Metric
            label="NORAD ID"
            value={chosen.id}
          />

          <Metric
            label="Latitude"
            value={chosen.latitude}
            unit="°"
          />

          <Metric
            label="Longitude"
            value={chosen.longitude}
            unit="°"
          />

          <Metric
            label="Altitude"
            value={chosen.altitude_km}
            unit="km"
          />

          <Metric
            label="Orbital-data age"
            value={chosen.orbital_age_hours}
            unit="hours"
          />

          <Metric
            label="Orbital data"
            value={
              chosen.stale
                ? "STALE"
                : "Within 7 days"
            }
          />

          <p className="muted">
            TLE epoch: {chosen.epoch}
          </p>
        </div>
      )}
    </Card>
  );

  // -------------------------------
  // Globe panel
  // -------------------------------

  const globePanel = (
    <Card title="3D Satellite Explorer">
      <p className="muted">
        Positions calculated from CelesTrak orbital
        elements. This is not a direct satellite
        connection.
      </p>

      <Globe
        satellites={satellites}
        selected={selected}
        onSelect={setSelected}
        station={station}
      />

      <button onClick={loadSatellites}>
        <RefreshCw size={15} />
        Refresh Positions
      </button>

      <p className="muted">
        Last orbital-data fetch:{" "}
        {source?.last_fetch || "Loading"}
      </p>
    </Card>
  );

  // -------------------------------
  // Ground station panel
  // -------------------------------

  const groundPanel = (
    <Card title="Ground Station and Visibility">
      <p className="muted">
        Default demonstration station: Bengaluru,
        India. Coordinates are editable.
      </p>

      <div className="form-grid">
        {Object.keys(station).map(key => (
          <label key={key}>
            {key.replaceAll("_", " ")}

            <input
              type="number"
              step="any"
              value={station[key]}
              onChange={e =>
                setStation({
                  ...station,
                  [key]: e.target.value
                })
              }
            />
          </label>
        ))}
      </div>

      <button
        onClick={checkVisibility}
        disabled={!selected || loading}
      >
        Calculate Visibility
      </button>

      {visibility && (
        <div className="details">
          <Metric
            label="Elevation"
            value={visibility.elevation_deg}
            unit="°"
          />

          <Metric
            label="Azimuth"
            value={visibility.azimuth_deg}
            unit="°"
          />

          <Metric
            label="Range"
            value={visibility.range_km}
            unit="km"
          />

          <Metric
            label="Geometric line of sight"
            value={
              visibility.geometric_line_of_sight
                ? "YES"
                : "NO"
            }
          />

          <Metric
            label="Above elevation mask"
            value={
              visibility.above_elevation_mask
                ? "YES"
                : "NO"
            }
          />

          <p className="muted">
            {visibility.note}
          </p>
        </div>
      )}
    </Card>
  );

  // -------------------------------
  // Monitoring panel
  // -------------------------------

  const monitoringPanel = (
    <>
      <Card title="Communication Link Monitoring">
        <p className="warning">
          SIMULATION MODE — These are generated
          values, not actual satellite-radio
          measurements.
        </p>

        <div className="metric-grid">
          <Metric
            label="Link State"
            value={metrics?.state}
          />

          <Metric
            label="Latency"
            value={metrics?.latency}
            unit="ms"
          />

          <Metric
            label="Packet Loss"
            value={metrics?.packet_loss}
            unit="%"
          />

          <Metric
            label="Jitter"
            value={metrics?.jitter}
            unit="ms"
          />

          <Metric
            label="Simulated SNR"
            value={metrics?.snr}
            unit="dB"
          />

          <Metric
            label="Simulated Throughput"
            value={metrics?.throughput}
            unit="Mbps"
          />

          <Metric
            label="Quality Score"
            value={metrics?.quality}
          />
        </div>

        <h4>Threshold Alerts</h4>

        {metrics?.violations?.length ? (
          metrics.violations.map((v, i) => (
            <p className="alert" key={i}>
              {v}
            </p>
          ))
        ) : (
          <p className="muted">
            No current threshold violations.
          </p>
        )}
      </Card>

      <div className="chart-grid">
        <Chart
          title="Latency (ms)"
          history={history}
          field="latency"
        />

        <Chart
          title="Packet Loss (%)"
          history={history}
          field="packet_loss"
          color="#f97316"
        />

        <Chart
          title="Simulated SNR (dB)"
          history={history}
          field="snr"
          color="#a78bfa"
        />

        <Chart
          title="Link Quality"
          history={history}
          field="quality"
          color="#22c55e"
        />
      </div>
    </>
  );

  // -------------------------------
  // Failure simulator panel
  // -------------------------------

  const simulatorPanel = (
    <Card title="Link Failure Simulator">
      <p className="muted">
        Reproducible experimental scenarios.
        No real radio interference is transmitted.
      </p>

      <label>
        Random seed

        <input
          type="number"
          min="0"
          max="1000000"
          value={seed}
          disabled={running}
          onChange={e => setSeed(e.target.value)}
        />
      </label>

      <div className="actions">
        <button onClick={startSimulation}>
          <Play size={16} />
          Start Simulation
        </button>

        <button
          className="danger"
          onClick={stopSimulation}
          disabled={!running}
        >
          <Square size={16} />
          Stop
        </button>
      </div>

      <h4>Select Failure Scenario</h4>

      <div className="scenario-grid">
        {SCENARIOS.map(item => (
          <button
            key={item}
            disabled={!running}
            className={
              scenario === item
                ? "active"
                : "secondary"
            }
            onClick={() => changeScenario(item)}
          >
            {item.replaceAll("_", " ").toUpperCase()}
          </button>
        ))}
      </div>

      <p className="muted">
        Running: {running ? "YES" : "NO"}
        {" | "}
        Scenario: {scenario}
      </p>
    </Card>
  );

  // -------------------------------
  // Threshold panel
  // -------------------------------

  const thresholdPanel = (
    <Card title="Acceptance Thresholds">
      <div className="form-grid">
        {Object.keys(thresholds).map(key => (
          <label key={key}>
            {key}

            <input
              type="number"
              step="any"
              value={thresholds[key]}
              onChange={e =>
                setThresholds({
                  ...thresholds,
                  [key]: e.target.value
                })
              }
            />
          </label>
        ))}
      </div>

      <button onClick={saveThresholds}>
        Save Thresholds
      </button>
    </Card>
  );

  // -------------------------------
  // ML analysis panel
  // -------------------------------

  const mlPanel = (
    <Card title="Machine Learning Analysis">
      <p className="warning">
        Models are initially trained on synthetic
        experimental data. Classification is not
        proof of real-world jamming.
      </p>

      <button
        onClick={runML}
        disabled={!metrics}
      >
        Run ML Prediction
      </button>

      {prediction && (
        <div className="details">
          <h4>Interference Classification</h4>

          <Metric
            label="Prediction"
            value={prediction.label}
          />

          <Metric
            label="Model Confidence"
            value={
              (
                prediction.confidence * 100
              ).toFixed(2)
            }
            unit="%"
          />

          <p className="muted">
            {prediction.warning}
          </p>
        </div>
      )}

      {linkPrediction && (
        <div className="details">
          <h4>Link-Quality Prediction</h4>

          <Metric
            label="Predicted Next Latency"
            value={
              linkPrediction.predicted_next_latency_ms
            }
            unit="ms"
          />

          <p className="muted">
            {linkPrediction.warning}
          </p>
        </div>
      )}

      {evaluation && (
        <div className="details">
          <h4>Saved Model Evaluation</h4>

          <Metric
            label="Classifier F1"
            value={
              evaluation.classification.f1.toFixed(3)
            }
          />

          <Metric
            label="Regression MAE"
            value={
              evaluation.regression.model_mae.toFixed(3)
            }
          />

          <Metric
            label="Baseline MAE"
            value={
              evaluation.regression.baseline_mae.toFixed(3)
            }
          />
        </div>
      )}
    </Card>
  );

  // -------------------------------
  // Experiment history
  // -------------------------------

  const historyPanel = (
    <Card title="Experiment History">
      <button onClick={loadEvents}>
        Refresh Events
      </button>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Scenario</th>
              <th>State</th>
              <th>Latency</th>
              <th>Loss</th>
              <th>Quality</th>
            </tr>
          </thead>

          <tbody>
            {events.map(event => (
              <tr key={event.id}>
                <td>{event.timestamp}</td>
                <td>{event.scenario}</td>
                <td>{event.state}</td>
                <td>{event.latency}</td>
                <td>{event.packet_loss}%</td>
                <td>{event.quality}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );

  // -------------------------------
  // Navigation
  // -------------------------------

  function renderPage() {
    switch (page) {
      case "Overview":
        return (
          <>
            <div className="metric-grid">
              <Metric
                label="Tracked Satellites"
                value={satellites.length}
              />

              <Metric
                label="Simulation"
                value={
                  running
                    ? "RUNNING"
                    : "STOPPED"
                }
              />

              <Metric
                label="Link State"
                value={metrics?.state}
              />

              <Metric
                label="Link Quality"
                value={metrics?.quality}
              />
            </div>

            {globePanel}
            {monitoringPanel}
          </>
        );

      case "Satellite Explorer":
        return (
          <>
            {satellitePanel}
            {globePanel}
          </>
        );

      case "Ground Stations":
        return (
          <>
            {satellitePanel}
            {groundPanel}
            {globePanel}
          </>
        );

      case "Link Monitoring":
        return (
          <>
            {monitoringPanel}
            {thresholdPanel}
          </>
        );

      case "Failure Simulator":
        return (
          <>
            {simulatorPanel}
            {monitoringPanel}
          </>
        );

      case "Jamming Detection":
      case "Link Prediction":
        return (
          <>
            {mlPanel}
            {monitoringPanel}
          </>
        );

      case "Experiment History":
        return historyPanel;

      case "About and Data Sources":
        return (
          <Card title="About the Project">
            <p>
              AI-Driven Satellite Network Monitoring
              and Communication Link Failure Prediction.
            </p>

            <p>
              Orbital data: CelesTrak.
              Orbital calculations: Skyfield.
              Simulated measurements: Local Python
              simulator. Machine learning: Scikit-learn.
            </p>

            <p className="warning">
              This application does not directly
              communicate with satellites or measure
              actual satellite RF signals.
            </p>
          </Card>
        );

      default:
        return null;
    }
  }

  // -------------------------------
  // Main dashboard layout
  // -------------------------------

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <Satellite size={28} />

          <div>
            <strong>SATELLITE AI</strong>
            <small>NETWORK OPERATIONS</small>
          </div>
        </div>

        <nav>
          {MENU.map(([name, Icon]) => (
            <button
              key={name}
              className={
                page === name
                  ? "nav active"
                  : "nav"
              }
              onClick={() => setPage(name)}
            >
              <Icon size={18} />
              {name}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          RESEARCH PROTOTYPE v1.0
        </div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <h1>{page}</h1>

            <p>
              Satellite Monitoring and Link
              Failure Prediction
            </p>
          </div>

          <span className="status">
            ●{" "}
            {running
              ? "SIMULATION ACTIVE"
              : "SYSTEM READY"}
          </span>
        </header>

        {error && (
          <div className="error">
            {error}

            <button onClick={() => setError("")}>
              Dismiss
            </button>
          </div>
        )}

        {renderPage()}
      </main>
    </div>
  );
}
