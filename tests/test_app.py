from unittest import TestCase

import pytest

from data5580_hw.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json == {"message": "Hello, Flask!"}


def test_create_user(client):
    response = client.post("/user", json={"email": "test@example.com", "name": "test"})
    assert response.status_code == 200


def test_get_user(client):
    payload = {"email": "test@example.com", "name": "test"}
    response = client.post("/user", json=payload)
    assert response.status_code == 200

    id = response.get_json(force=True)['id']
    assert id is not None

    payload['id'] = id
    response = client.get("/user/{}".format(id))
    TestCase().assertDictEqual(response.get_json(force=True), payload)


def test_update_user(client):
    payload = {"email": "test@example.com", "name": "test"}
    response = client.post("/user", json=payload)
    assert response.status_code == 200

    id = response.get_json(force=True)['id']
    assert id is not None

    payload = {"email": "test@example.com", "name": "test2"}
    response = client.patch("/user/{}".format(id), json=payload)
    assert response.status_code == 200

    payload['id'] = id
    response = client.get("/user/{}".format(id))
    TestCase().assertDictEqual(response.get_json(force=True), payload)

    assert response.get_json(force=True)["name"] == "test2"
