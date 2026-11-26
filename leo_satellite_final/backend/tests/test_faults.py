from fastapi.testclient import TestClient
from app.main import app, faults, state
from app.simulator import snapshot

def test_fault_injection_lifecycle():
    with TestClient(app) as client:
        client.post('/api/v1/reset')
        snap=client.get('/api/v1/snapshot').json()
        assert snap['links']
        link_id=snap['links'][0]['id']
        created=client.post('/api/v1/faults',json={'target':link_id,'kind':'outage','severity':100,'duration_s':60})
        assert created.status_code==200
        fault=created.json()['fault']
        updated=client.get('/api/v1/snapshot').json()
        affected=next(l for l in updated['links'] if l['id']==link_id)
        assert affected['state']=='failed'
        assert affected['loss_pct']==100
        assert client.get('/api/v1/faults').json()['faults']
        assert client.delete('/api/v1/faults/'+fault['id']).status_code==200
        assert client.get('/api/v1/faults').json()['faults']==[]
        assert client.post('/api/v1/faults',json={'target':'NOT-A-LINK','kind':'outage'}).status_code==422

def test_scheduled_fault():
    with TestClient(app) as client:
        client.post('/api/v1/reset')
        link=client.get('/api/v1/snapshot').json()['links'][0]['id']
        client.post('/api/v1/faults',json={'target':link,'kind':'overload','severity':80,'delay_s':20,'duration_s':30})
        assert client.get('/api/v1/faults').json()['faults'][0]['status']=='scheduled'
        client.delete('/api/v1/faults')
