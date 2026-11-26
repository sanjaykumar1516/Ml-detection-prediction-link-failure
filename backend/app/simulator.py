"""Deterministic synthetic constellation and simplified telemetry; not real satellite data."""
import math
import random
import networkx as nx

EARTH_KM = 6371.0
ALT_KM = 550.0
RADIUS = EARTH_KM + ALT_KM
PERIOD_S = 2 * math.pi * math.sqrt((RADIUS * 1000) ** 3 / 3.986004418e14)


def position(satellite, t):
    plane, slot = divmod(satellite, 4)
    inclination = math.radians(53)
    raan = 2 * math.pi * plane / 3
    angle = 2 * math.pi * (slot / 4 + t / PERIOD_S) + plane * 0.24
    x = RADIUS * (math.cos(raan) * math.cos(angle) - math.sin(raan) * math.sin(angle) * math.cos(inclination))
    y = RADIUS * (math.sin(raan) * math.cos(angle) + math.cos(raan) * math.sin(angle) * math.cos(inclination))
    z = RADIUS * math.sin(angle) * math.sin(inclination)
    return (x, y, z)


def visible(a, b):
    delta = [b[i] - a[i] for i in range(3)]
    norm = sum(v*v for v in delta)
    if norm == 0:
        return False
    u = max(0, min(1, -sum(a[i]*delta[i] for i in range(3))/norm))
    closest = [a[i] + u*delta[i] for i in range(3)]
    return sum(v*v for v in closest) > EARTH_KM**2


def snapshot(t, scenario="healthy", seed=42, faults=None, mesh=False):
    points = {f"SAT-{i+1:02d}": position(i, t) for i in range(12)}
    nodes = [{"id": k, "position": list(v), "kind": "satellite"} for k, v in points.items()]
    # Two fixed ECEF ground sites; approximate latitude/longitude conversion.
    for name, lat, lon in [("GS-Bengaluru", 12.97, 77.59), ("GS-Singapore", 1.35, 103.82), ("GS-Frankfurt", 50.11, 8.68), ("GS-Sydney", -33.87, 151.21)]:
        la, lo = math.radians(lat), math.radians(lon)
        points[name] = (EARTH_KM*math.cos(la)*math.cos(lo), EARTH_KM*math.cos(la)*math.sin(lo), EARTH_KM*math.sin(la))
        nodes.append({"id": name, "position": list(points[name]), "kind": "ground"})
    links = []
    ids = list(points)
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            if a.startswith("GS") and b.startswith("GS"):
                continue
            pa, pb = points[a], points[b]
            distance = math.dist(pa, pb)
            if a.startswith("GS") or b.startswith("GS"):
                ground = pa if a.startswith("GS") else pb
                sat = pb if a.startswith("GS") else pa
                delta = [sat[j]-ground[j] for j in range(3)]
                elevation = sum(delta[j]*ground[j] for j in range(3)) / (math.sqrt(sum(x*x for x in delta))*EARTH_KM)
                allowed = mesh or elevation > math.sin(math.radians(10))
            else:
                allowed = mesh or (visible(pa, pb) and distance < 4800)
            if not allowed:
                continue
            key = f"{a}|{b}"
            rng = random.Random(f"{seed}-{key}-{int(t//5)}")
            interference = rng.random() * 0.8
            if mesh and a.startswith("SAT") and b.startswith("SAT"):
                distance = min(distance, 3500)  # abstract mesh telemetry, not physical distance
            # The scenario selects a deterministic available inter-satellite edge later.
            snr = max(0.5, 24 - distance/500 - interference)
            packet_loss = max(0.0, (12-snr)*1.4)
            latency = distance/299.792458 + 3 + rng.random()*2
            links.append({"id": key, "source": a, "target": b, "distance_km": round(distance,2),
                          "snr_db": round(snr,2), "rssi_dbm": round(-48-distance/350,2),
                          "ber": round(0.5*math.erfc(math.sqrt(10**(snr/10))),10),
                          "loss_pct": round(packet_loss,2), "latency_ms": round(latency,2),
                          "jitter_ms": round(0.2+rng.random()*2,2), "bandwidth_mbps": round(max(1,150-distance/55),2),
                          "doppler_hz": round(1000*math.sin(t/100+i),2), "state": "healthy"})
    candidates = [l for l in links if l["source"].startswith("SAT") and l["target"].startswith("SAT")]
    target = sorted(candidates, key=lambda x:x["id"])[0] if candidates else None
    if target and scenario != "healthy":
        phase = t % 180
        if scenario == "degradation":
            severity = min(1, phase/90)
        elif scenario == "outage":
            severity = 1 if 45 <= phase < 120 else 0
        elif scenario == "interference":
            severity = 0.7
        else:
            severity = 0
        target["snr_db"] = round(max(0.2,target["snr_db"]-20*severity),2)
        target["loss_pct"] = round(min(100,target["loss_pct"]+100*severity),2)
        target["latency_ms"] = round(target["latency_ms"]+180*severity,2)
        target["bandwidth_mbps"] = round(max(0.1,target["bandwidth_mbps"]*(1-severity)),2)
        target["state"] = "failed" if severity >= 0.95 else "degraded" if severity > 0.2 else "healthy"
        target["fault_injected"] = True
    # Explicit operator-injected faults override scenario telemetry, without altering geometry.
    for fault in (faults or []):
        if not (fault["start_s"] <= t < fault["end_s"]):
            continue
        affected = [l for l in links if fault["target"] == "ALL" or l["id"] == fault["target"]]
        for l in affected:
            severity = fault["severity"] / 100.0
            kind = fault["kind"]
            profile = fault.get("profile", "sudden")
            if profile == "gradual": severity *= min(1, (t-fault["start_s"])/max(1,(fault["end_s"]-fault["start_s"])*0.6))
            if profile == "intermittent" and int((t-fault["start_s"])/3)%2: continue
            kind = {"climate":"interference","solar":"interference","intrusion":"degradation","jamming":"interference","ddos":"overload","antenna":"degradation","power":"hardware","handover":"degradation"}.get(kind,kind)
            if kind == "outage" and profile == "sudden":
                l["snr_db"] = 0.0
                l["loss_pct"] = 100.0
                l["bandwidth_mbps"] = 0.0
                l["state"] = "failed"
            elif kind == "overload":
                # Synthetic utilization and capacity/queue impact; no packet-level traffic model.
                l["utilization_pct"] = round(65 + 135 * severity, 1)
                l["loss_pct"] = round(min(100, l["loss_pct"] + 80 * severity), 2)
                l["latency_ms"] = round(l["latency_ms"] + 250 * severity, 2)
                l["jitter_ms"] = round(l["jitter_ms"] + 30 * severity, 2)
                l["bandwidth_mbps"] = round(max(0.1, l["bandwidth_mbps"] * (1 - 0.9 * severity)), 2)
                l["state"] = "failed" if severity >= 0.95 else "degraded"
            elif kind in ("interference", "degradation", "outage", "hardware") and (kind != "hardware" or profile != "sudden"):
                l["snr_db"] = round(max(0.1, l["snr_db"] - 25 * severity), 2)
                l["loss_pct"] = round(min(100, l["loss_pct"] + 95 * severity), 2)
                l["latency_ms"] = round(l["latency_ms"] + 120 * severity, 2)
                l["bandwidth_mbps"] = round(max(0.1, l["bandwidth_mbps"] * (1 - severity)), 2)
                l["state"] = "failed" if severity >= 0.95 else "degraded" if severity > 0.2 else l["state"]
            elif kind == "hardware":
                l["loss_pct"] = 100.0
                l["bandwidth_mbps"] = 0.0
                l["state"] = "failed"
            l["fault_injected"] = True
            l["manual_fault_id"] = fault["id"]
            l["fault_type"] = fault["kind"]
    for l in links:
        if l["state"] == "healthy" and (l["snr_db"] < 12 or l["loss_pct"] > 10):
            l["state"] = "degraded"
    return {"time_s":t, "scenario":scenario, "nodes":nodes, "links":links,
            "injected_link":target["id"] if target else None}


def routes(snap, source, target):
    graph = nx.Graph()
    graph.add_nodes_from(n["id"] for n in snap["nodes"])
    for link in snap["links"]:
        if link["state"] == "failed":
            continue
        graph.add_edge(link["source"],link["target"], latency=link["latency_ms"],
                       current=link["latency_ms"]*(1+link["loss_pct"]/20),
                       predictive=link["latency_ms"]*(1+link["loss_pct"]/8 + max(0,18-link["snr_db"])/8))
    result={}
    for name, weight in [("shortest",None),("current","current"),("predictive","predictive")]:
        try:
            result[name] = nx.shortest_path(graph,source,target,weight=weight)
        except (nx.NetworkXNoPath,nx.NodeNotFound):
            result[name]=[]
    return result
