from fastapi.testclient import TestClient
from app.main import app
from app.simulator import snapshot, position, visible, routes

def test_health():
    with TestClient(app) as client:
        assert client.get('/api/v1/health').status_code==200
        assert client.get('/api/v1/snapshot').json()['nodes'].__len__()==16
        assert client.get('/api/v1/routes').status_code==200
        assert client.post('/api/v1/simulation',json={'scenario':'invalid'}).status_code==422

def test_geometry():
    assert len(position(0,0))==3
    assert visible((7000,0,0),(-7000,0,0)) is False
    assert len(snapshot(0)['nodes'])==16
