# Fault Injection Lab

The new sidebar page injects manual faults into active satellite links. Supported types: outage, overload, interference, degradation, hardware. The form accepts a target, severity, start delay and duration in **simulation seconds**. The backend provides `GET/POST/DELETE /api/v1/faults` and `DELETE /api/v1/faults/{id}`. `POST /api/v1/reset` clears all faults.

Run: `docker compose up --build -d`; open `http://localhost:5173` and select **Fault Injection Lab**. Use the page to select a live link, select **Complete outage**, set start delay 0 and duration 60, click **Inject Fault Now**, then open **Overview** to see the failed-link count.

The selected link must still be geometrically visible. Faults are held in process memory and reset on backend restart; creation/stop events are saved in SQLite. Overload is a telemetry-level synthetic approximation, not real queueing/packet traffic; there is no automatic cascading traffic propagation in this version. Fault state is simulation ground truth, not an AI prediction.
