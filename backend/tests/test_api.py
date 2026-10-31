import pytest
from fastapi.testclient import TestClient

from heatsplat.export import export_scene
from heatsplat.synth import build_plane


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("scenes")
    export_scene(build_plane(), root / "plane", {"name": "Test plane"})
    (root / "broken").mkdir()  # a directory without scene files must be ignored
    half = root / "half-exported"  # JSON present but no splat/npz: must be ignored too
    half.mkdir()
    for name in ("meta.json", "leaks.json"):
        (half / name).write_text((root / "plane" / name).read_text())
    import os

    os.environ["HEATSPLAT_DATA"] = str(root)
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    os.environ.pop("HEATSPLAT_DATA", None)


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_list_scenes_skips_incomplete_dirs(client):
    scenes = client.get("/api/scenes").json()
    assert [s["scene_id"] for s in scenes] == ["plane"]
    assert scenes[0]["n_leaks"] == 1 and scenes[0]["name"] == "Test plane"


def test_scene_metadata_and_404(client):
    assert client.get("/api/scenes/plane").json()["name"] == "Test plane"
    assert client.get("/api/scenes/nope").status_code == 404
    assert client.get("/api/scenes/..").status_code in (400, 404)


def test_splat_download_per_mode(client):
    for mode in ("rgb", "thermal", "leaks"):
        r = client.get(f"/api/scenes/plane/splat", params={"mode": mode})
        assert r.status_code == 200 and len(r.content) % 32 == 0 and len(r.content) > 0
    assert client.get("/api/scenes/plane/splat", params={"mode": "xray"}).status_code == 422


def test_leaks_and_live_reanalysis(client):
    before = client.get("/api/scenes/plane/leaks").json()
    assert before["summary"]["n_leaks"] == 1
    strict = client.post("/api/scenes/plane/analyze", json={"min_delta_c": 10.0}).json()
    assert strict["summary"]["n_leaks"] == 0
    assert client.get("/api/scenes/plane/leaks").json()["params"]["min_delta_c"] == 10.0
    restored = client.post("/api/scenes/plane/analyze", json={"min_delta_c": 1.5, "price_per_kwh": 0.30}).json()
    assert restored["summary"]["n_leaks"] == 1
    assert restored["cost_model"]["price_per_kwh"] == 0.30


def test_analyze_rejects_bad_values(client):
    assert client.post("/api/scenes/plane/analyze", json={"min_delta_c": -1}).status_code == 422
    assert client.post("/api/scenes/plane/analyze", json={"sign": 2}).status_code == 422
    assert client.post("/api/scenes/plane/analyze", content='{"min_delta_c": NaN}', headers={"Content-Type": "application/json"}).status_code == 422
    assert client.get("/api/scenes/" + "a" * 500).status_code in (400, 404)


def test_splat_responses_are_revalidated(client):
    assert client.get("/api/scenes/plane/splat", params={"mode": "leaks"}).headers["cache-control"] == "no-cache"


def test_report_markdown(client):
    r = client.post("/api/scenes/plane/report").json()
    assert r["generator"] == "template-v1"
    assert "# Thermal envelope audit" in r["markdown"]
    assert "Recommended fix" in r["markdown"]
