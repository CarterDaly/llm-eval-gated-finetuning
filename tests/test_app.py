import pytest

from data5580_hw.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def _create_user(client, name="Test User", email="test@example.com"):
    response = client.post("/users", json={"name": name, "email": email})
    assert response.status_code == 201
    return response.get_json()


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json() == {"message": "Hello, Flask!"}


def test_create_user_success(client):
    response = client.post("/users", json={"email": "test@example.com", "name": "test"})
    assert response.status_code == 201
    body = response.get_json()
    assert body["message"] == "User created"
    assert body["user"]["name"] == "test"
    assert body["user"]["email"] == "test@example.com"
    assert len(body["user"]["id"]) == 32
    int(body["user"]["id"], 16)


def test_create_user_ignores_client_id(client):
    response = client.post(
        "/users",
        json={"id": "client-id", "name": "Client", "email": "client@example.com"},
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["user"]["id"] != "client-id"


def test_create_user_duplicate_email(client):
    _create_user(client, email="dup@example.com")
    response = client.post("/users", json={"name": "dup", "email": "dup@example.com"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Email is already in use"


def test_create_user_missing_fields(client):
    response = client.post("/users", json={"name": "only-name"})
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "Missing required fields" in error
    assert "email" in error


def test_create_user_invalid_email(client):
    response = client.post("/users", json={"name": "bad", "email": "not-an-email"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid email format"


def test_get_user_success(client):
    created = _create_user(client, name="A", email="a@example.com")
    user_id = created["user"]["id"]
    response = client.get(f"/users/{user_id}")
    assert response.status_code == 200
    body = response.get_json()
    assert body["user"]["id"] == user_id
    assert body["user"]["name"] == "A"
    assert body["user"]["email"] == "a@example.com"


def test_get_user_not_found(client):
    response = client.get("/users/missing")
    assert response.status_code == 404
    assert response.get_json()["error"] == "User not found"


def test_update_user_patch_success(client):
    created = _create_user(client, name="Old", email="old@example.com")
    user_id = created["user"]["id"]
    response = client.patch(f"/users/{user_id}", json={"name": "New"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["message"] == "User updated"
    assert body["user"]["name"] == "New"
    assert body["user"]["email"] == "old@example.com"


def test_update_user_patch_missing_fields(client):
    created = _create_user(client, name="Old", email="old@example.com")
    user_id = created["user"]["id"]
    response = client.patch(f"/users/{user_id}", json={})
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "Missing required fields" in error


def test_update_user_patch_not_found(client):
    response = client.patch("/users/missing", json={"name": "New"})
    assert response.status_code == 404
    assert response.get_json()["error"] == "User not found"


def test_update_user_patch_duplicate_email(client):
    first = _create_user(client, name="First", email="first@example.com")
    second = _create_user(client, name="Second", email="second@example.com")
    response = client.patch(f"/users/{second['user']['id']}", json={"email": first["user"]["email"]})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Email is already in use"


def test_replace_user_put_success(client):
    created = _create_user(client, name="Old", email="old@example.com")
    user_id = created["user"]["id"]
    response = client.put(f"/users/{user_id}", json={"name": "New", "email": "new@example.com"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["message"] == "User details updated successfully."
    assert body["user"]["name"] == "New"
    assert body["user"]["email"] == "new@example.com"


def test_replace_user_put_missing_fields(client):
    created = _create_user(client, name="Old", email="old@example.com")
    user_id = created["user"]["id"]
    response = client.put(f"/users/{user_id}", json={"name": "New"})
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "Missing required fields" in error
    assert "email" in error


def test_replace_user_put_not_found(client):
    response = client.put("/users/missing", json={"name": "New", "email": "new@example.com"})
    assert response.status_code == 404
    assert response.get_json()["error"] == "User not found"


def test_delete_user_success(client):
    created = _create_user(client, name="Old", email="old@example.com")
    user_id = created["user"]["id"]
    response = client.delete(f"/users/{user_id}")
    assert response.status_code == 200
    assert response.get_json()["message"] == "User deleted"

    response = client.get(f"/users/{user_id}")
    assert response.status_code == 404


def test_delete_user_not_found(client):
    response = client.delete("/users/missing")
    assert response.status_code == 404
    assert response.get_json()["error"] == "User not found"


def test_list_users(client):
    _create_user(client, name="First", email="first@example.com")
    _create_user(client, name="Second", email="second@example.com")
    response = client.get("/users")
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["users"]) == 2
    emails = {user["email"] for user in body["users"]}
    assert emails == {"first@example.com", "second@example.com"}
