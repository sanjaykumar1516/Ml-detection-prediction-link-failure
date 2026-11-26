import asyncio
import json
import uuid
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.simulator import snapshot, routes
from app.ml import models, infer

DB="data/events.sqlite3"
faults=[]
schedules=[]
mesh=True
state={"running":True,"scenario":"healthy","speed":1.0,"sim_time":0.0,"last_wall":time.monotonic()}

def db():
    os.makedirs("data",exist_ok=True)
    con=sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, sim_time REAL, kind TEXT, details TEXT)")
    con.commit()
    return con

def log(kind,details):
    with db() as con:
        con.execute("INSERT INTO events(sim_time,kind,details) VALUES(?,?,?)",(state["sim_time"],kind,json.dumps(details)))

def tick():
    now=time.monotonic()
    if state["running"]:
        state["sim_time"] += min(2,now-state["last_wall"])*state["speed"]
    state["last_wall"]=now
    snap={**snapshot(round(state["sim_time"],2),state["scenario"],faults=faults,mesh=mesh), "running":state["running"], "speed":state["speed"], "mesh":mesh}
    if state['running']:
        process_schedules(snap)
    return snap

@asynccontextmanager
async def lifespan(app):
    con=db();con.close()
    async def clock():
        while True:
            tick()
            await asyncio.sleep(0.25)
    task=asyncio.create_task(clock())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)

app=FastAPI(title="LEO Satellite AI - Synthetic Research Prototype",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],allow_methods=["*"],allow_headers=["*"])

class FaultRequest(BaseModel):
    target:str=Field(min_length=1)
    kind:str
    severity:int=Field(default=80,ge=1,le=100)
    delay_s:float=Field(default=0,ge=0,le=86400)
    profile:str="sudden"
    duration_s:float=Field(default=60,gt=0,le=86400)

class Config(BaseModel):
    mesh:bool=True
    running:bool=True
    speed:float=Field(1,ge=0.1,le=100)
    scenario:str="healthy"

@app.get("/api/v1/health")
async def health():
    return {"status":"healthy","simulation":"synthetic","satellites":12,"ground_stations":4}

@app.get("/api/v1/snapshot")
async def get_snapshot():
    return tick()

@app.post("/api/v1/simulation")
async def configure(config:Config):
    global mesh
    if config.scenario not in ("healthy","degradation","outage","interference"):
        raise HTTPException(422,"Unknown scenario")
    tick()
    mesh=config.mesh
    state.update(running=config.running,speed=config.speed,scenario=config.scenario)
    log("configuration",config.model_dump())
    return {"ok":True,"state":state}

@app.get("/api/v1/faults")
async def list_faults():
    snap=tick()
    for fault in faults:
        fault["status"] = "scheduled" if snap["time_s"] < fault["start_s"] else "active" if snap["time_s"] < fault["end_s"] else "completed"
    return {"faults":faults,"time_s":snap["time_s"]}

@app.post("/api/v1/faults")
async def create_fault(req:FaultRequest):
    if req.kind not in ("outage","overload","interference","degradation","hardware","climate","solar","intrusion","jamming","ddos","antenna","power","handover"):
        raise HTTPException(422,"Unsupported fault type")
    if req.profile not in ("sudden","gradual","intermittent"): raise HTTPException(422,"Invalid profile")
    snap=tick()
    if req.target != "ALL" and req.target not in {l["id"] for l in snap["links"]}:
        raise HTTPException(422,"Select an active link from the current snapshot")
    if len(faults) >= 200:
        raise HTTPException(409,"Fault limit reached; clear fault history")
    fault={"id":uuid.uuid4().hex[:12],"target":req.target,"kind":req.kind,
           "severity":req.severity,"profile":req.profile,"start_s":round(state["sim_time"]+req.delay_s,2),
           "end_s":round(state["sim_time"]+req.delay_s+req.duration_s,2)}
    faults.append(fault)
    log("fault_created",fault)
    return {"ok":True,"fault":fault}

@app.delete("/api/v1/faults/{fault_id}")
async def stop_fault(fault_id:str):
    for fault in faults:
        if fault["id"] == fault_id:
            faults.remove(fault)
            log("fault_stopped",{"id":fault_id})
            return {"ok":True}
    raise HTTPException(404,"Fault not found")

@app.delete("/api/v1/faults")
async def clear_faults():
    count=len(faults)
    faults.clear()
    log("faults_cleared",{"count":count})
    return {"ok":True,"cleared":count}

@app.post("/api/v1/reset")
async def reset():
    faults.clear()
    schedules.clear()
    state.update(sim_time=0.0,scenario="healthy",running=True,speed=1.0,last_wall=time.monotonic())
    log("reset",{})
    return {"ok":True}

@app.get("/api/v1/links/{link_id}/prediction")
async def predict(link_id:str):
    snap=tick()
    link=next((l for l in snap["links"] if l["id"]==link_id),None)
    if not link:raise HTTPException(404,"Link not currently visible")
    return {"link_id":link_id,"result":infer(link),"note":"Synthetic baseline prediction, not validated future-event prediction"}

@app.get("/api/v1/routes")
async def get_routes(source:str="SAT-01",target:str="SAT-07"):
    snap=tick()
    return {"source":source,"target":target,"routes":routes(snap,source,target)}

@app.get("/api/v1/evaluation")
async def evaluation():
    return models()[4]

@app.get("/api/v1/events")
async def events():
    with db() as con:
        rows=con.execute("SELECT id,sim_time,kind,details FROM events ORDER BY id DESC LIMIT 100").fetchall()
    return [{"id":r[0],"time_s":r[1],"kind":r[2],"details":json.loads(r[3])} for r in rows]

@app.websocket("/ws")
async def ws(socket:WebSocket):
    await socket.accept()
    try:
        while True:
            await socket.send_json(tick())
            await asyncio.sleep(1)
    except (WebSocketDisconnect,RuntimeError):
        pass

@app.get('/api/v1/research/cascade')
async def cascade_research(seed:int=500):
    from app.research import report, routing_trial
    return {**report(seed), "routing_trial": routing_trial(seed)}

@app.get('/api/v1/research/benchmark')
async def benchmark_research(seed:int=500):
    from app.research import benchmark
    return benchmark(seed)

class MessageRequest(BaseModel):
    source:str
    target:str
    message:str=Field(min_length=1,max_length=4000)
    traffic:str='Normal data'

@app.get('/api/v1/messages')
async def message_history():
    with db() as con:
        rows=con.execute("SELECT details FROM events WHERE kind='message' ORDER BY id DESC LIMIT 100").fetchall()
    return [json.loads(r[0]) for r in rows]

@app.post('/api/v1/messages')
async def transmit(req:MessageRequest):
    return deliver(req,tick())

def deliver(req:MessageRequest,snap,schedule_id=None):
    import networkx as nx
    import random
    ids={n['id'] for n in snap['nodes']}
    if req.source not in ids or req.target not in ids or req.source==req.target or not req.message.strip():
        raise HTTPException(422,'Select distinct valid endpoints and a message')
    demand={'Emergency communication':2,'Satellite control commands':1,'Normal data':8}.get(req.traffic)
    if demand is None: raise HTTPException(422,'Unknown traffic class')
    graph=nx.Graph();graph.add_nodes_from(ids)
    for l in snap['links']:
        if l['state']=='failed' or l['bandwidth_mbps']<demand: continue
        risk=max(0,(18-l['snr_db'])/18)+l['loss_pct']/100
        graph.add_edge(l['source'],l['target'],latency=l['latency_ms'],risk=l['latency_ms']/50+2*risk,loss=l['loss_pct']/100)
    results={}; draws={frozenset((a,b)):random.random() for a,b in graph.edges}
    for name,weight in [('Reactive','latency'),('Risk-aware','risk')]:
        try: path=nx.shortest_path(graph,req.source,req.target,weight=weight)
        except nx.NetworkXNoPath: path=[]
        edges=list(zip(path,path[1:])); delivered=bool(path) and all(draws[frozenset((a,b))]>=graph[a][b]['loss'] for a,b in edges)
        results[name]={'path':path,'delivered':delivered,'latency_ms':round(sum(graph[a][b]['latency'] for a,b in edges),2) if path else None}
    record={**req.model_dump(),'id':uuid.uuid4().hex,'time_s':snap['time_s'],'results':results,'schedule_id':schedule_id}
    log('message',record)
    return record


class ScheduleRequest(MessageRequest):
    delay_s:float=Field(default=0,ge=0,le=86400)
    interval_s:float=Field(default=5,ge=1,le=86400)
    count:int=Field(default=1,ge=1,le=1000)

@app.post('/api/v1/message-schedules')
async def schedule_message(req:ScheduleRequest):
    snap=tick()
    ids={n['id'] for n in snap['nodes']}
    if req.source not in ids or req.target not in ids or req.source==req.target or not req.message.strip() or req.traffic not in ('Emergency communication','Satellite control commands','Normal data'):
        raise HTTPException(422,'Invalid communication configuration')
    if len(schedules)>=200: raise HTTPException(409,'Schedule limit reached; reset simulation')
    job={**req.model_dump(),'id':uuid.uuid4().hex[:12],'next_s':snap['time_s']+req.delay_s,'sent':0,'status':'scheduled'}
    schedules.append(job)
    log('communication_scheduled',job)
    return job

@app.get('/api/v1/message-schedules')
async def list_schedules():
    return schedules

@app.delete('/api/v1/message-schedules/{job_id}')
async def cancel_schedule(job_id:str):
    for job in schedules:
        if job['id']==job_id:
            if job['status'] not in ('completed','cancelled'):
                job['status']='cancelled'
                log('communication_cancelled',{'id':job_id})
            return job
    raise HTTPException(404,'Schedule not found')

def process_schedules(snap):
    # Claim each due slot before delivery. All tick callers run on the event loop.
    for job in sorted(schedules,key=lambda j:({'Emergency communication':0,'Satellite control commands':1,'Normal data':2}[j['traffic']],j['next_s'])):
        while job['status'] in ('scheduled','active') and job['next_s']<=snap['time_s']:
            due=job['next_s']
            job['sent']+=1
            job['next_s']+=job['interval_s']
            job['status']='completed' if job['sent']>=job['count'] else 'active'
            # Evaluate at the scheduled simulation time, not the later polling time.
            frame=snapshot(due,state['scenario'],faults=faults,mesh=mesh)
            deliver(MessageRequest(**job),frame,job['id'])
