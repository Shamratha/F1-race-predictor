"""Endpoint tests via FastAPI's TestClient."""


def test_meta_endpoint(client):
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body and "roc_auc" in body["metrics"]
    assert isinstance(body["seasons"], list) and body["seasons"]


def test_races_endpoint(client):
    r = client.get("/api/races")
    assert r.status_code == 200
    races = r.json()["races"]
    assert races                                  # non-empty
    # keys are season strings/ints, values are lists of circuits
    some_season = next(iter(races))
    assert isinstance(races[some_season], list) and races[some_season]


def test_predict_valid_race(client):
    races = client.get("/api/races").json()["races"]
    season = max(int(s) for s in races)
    circuit = races[str(season)][0] if str(season) in races else races[season][0]
    r = client.get("/api/predict", params={"season": season, "circuit": circuit})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predicted_podium"]) == 3
    assert len(body["drivers"]) >= 10
    assert body["race"]["changed"] is False


def test_predict_invalid_race_returns_404(client):
    r = client.get("/api/predict", params={"season": 1999, "circuit": "Nowhere GP"})
    assert r.status_code == 404
    assert "error" in r.json()


def test_predict_whatif_penalty_changes_result(client):
    races = client.get("/api/races").json()["races"]
    season = max(int(s) for s in races)
    circuit = races[str(season)][0] if str(season) in races else races[season][0]
    base = client.get("/api/predict", params={"season": season, "circuit": circuit}).json()
    favourite = base["predicted_podium"][0]
    r = client.get("/api/predict", params={
        "season": season, "circuit": circuit,
        "penalty_driver": favourite, "penalty_grid": 20,
    })
    assert r.status_code == 200
    assert r.json()["race"]["changed"] is True


def test_weather_live_gracefully_unavailable(client):
    # no OPENWEATHER_API_KEY in the test environment
    r = client.get("/api/weather/live", params={"circuit": "Monaco Grand Prix"})
    assert r.status_code == 200
    assert r.json()["source"] == "unavailable"
