# LEO Satellite AI — Docker Research Prototype

## Important status
This is a **working prototype**, not the fully validated research system in the specification. It implements a synthetic constellation, simplified telemetry, fault scenarios, baseline ML models, calibrated *proxy* horizon labels, route comparisons, WebSocket monitoring, a React dashboard and SQLite configuration history.

**Not yet implemented:** LSTM, real event-horizon training labels derived from time series, scenario-level leakage-safe evaluation, actual predictive routing with hysteresis, Cesium 3D, traffic delivery simulation, ablation experiments and a defensible measured comparison of network availability. Do not claim these features are finished or claim superiority.

## Requirements
- Windows 10/11 with Docker Desktop and Linux containers enabled
- Docker Compose v2
- No local Python or Node.js installation required

## Start
In PowerShell, open the directory containing `docker-compose.yml`:

```powershell
docker compose down --remove-orphans
docker compose up --build -d
docker compose ps
```

Frontend: http://localhost:5173
Backend API: http://localhost:8000/docs
Health: http://localhost:8000/api/v1/health

On first use, Model Evaluation may take time while models train in the backend container. The models are held in memory and retrained on backend restart. These metrics are only synthetic proxy baselines.

## Test

```powershell
docker compose exec backend python -m pytest -v
docker compose exec frontend npm run build
docker compose logs -f backend frontend
```

## Fix Windows bind-mount Vite EACCES
`frontend_node_modules` is a named Docker volume. This prevents Vite from renaming cache files in a Windows-hosted node_modules folder. If old cache remains, use:

```powershell
docker compose down
docker compose up --build -d
```

If dependencies change, rebuild the frontend volume selectively (the command below resets all Compose volumes, including SQLite data, so back up data first):

```powershell
docker compose down -v
docker compose up --build -d
```

## API
- `GET /api/v1/health`
- `GET /api/v1/snapshot`
- `POST /api/v1/simulation` with `{"running":true,"speed":1,"scenario":"degradation"}`
- `POST /api/v1/reset`
- `GET /api/v1/links/{link_id}/prediction` (URL-encode the pipe in link IDs)
- `GET /api/v1/routes?source=SAT-01&target=SAT-07`
- `GET /api/v1/evaluation`
- `GET /api/v1/events`
- `WS /ws`

## Assumptions
12 satellites in 3 circular orbital planes at 550 km altitude, inclination 53 degrees; circular two-body angular velocity; no J2, drag or TLE input. Earth spherical radius 6371 km. Satellite-satellite links require unobstructed Earth line of sight and distance under 4800 km; ground-station links require elevation above 10 degrees. Telemetry is a simplified function of distance, scenario severity and deterministic noise, **not an RF link budget**. Doppler is illustrative, not derived from range rate. Simulated clock runs at configurable speed.

## Research caution
The prediction dataset uses IID generated features with proxy labels, not simulated link time-series with verified 15/30/60-second event windows. Calibration is fitted on separate random-seed validation data. Metrics should not be interpreted as a scientific demonstration of early warning or operational satellite performance.

## Mesh research dashboard upgrade
Run `docker compose up --build -d` and open http://localhost:5173. Three dashboard sections retain the original API and model evaluation. Full mesh is an abstract 66-ISL topology; distance used for mesh telemetry is capped, not physical. The paired routing view compares latency-only and telemetry-risk-weighted Dijkstra on the same snapshots. It does not use calibrated ML predictions for routing yet. Traffic classes use illustrative minimum bandwidth thresholds, not actual priority queues. Replay is a browser-local rolling buffer of 600 frames while Mission Control is mounted, not persistent recording. Link health index weights healthy links 1 and degraded links 0.5. Fault reasons map to simplified telemetry effects; climate is intended for ground links. Fault timing is relative to current simulation time. Existing ML metrics remain synthetic proxy results.

## Experimental graph research
Fault & Research Lab now includes a separate seeded load-cascade benchmark, temporal graph message-passing classifier, held-out metrics, residual uncertainty score and contextual-bandit routing comparison. JSON export contains 6,600 edge-time records per seed. The four scenario names currently share the same cascade mechanism. This is not integrated into live routing; PPO, conformal coverage, realistic weather models and physical validation remain future work. Raw JSON evaluation and event displays are replaced by structured metric panels. CPU PyTorch is required. The original folder layout is retained with one additional backend module.
