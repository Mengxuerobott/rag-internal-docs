"""
tests/test_rbac.py
──────────────────
Tests for the RBAC layer: role expansion, permission stamping, JWT flow,
and end-to-end API enforcement (forbidden chunks must never appear).

Run:
    pytest tests/test_rbac.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth.rbac import (
    ALL_ROLES,
    FOLDER_PERMISSIONS,
    can_access,
    expand_roles,
    get_allowed_roles_for_path,
)
from auth.jwt_handler import authenticate_user, create_access_token, decode_token


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Role expansion tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpandRoles:
    def test_employee_only_sees_employee(self):
        assert expand_roles("employee") == ["employee"]

    def test_hr_sees_employee_and_hr(self):
        result = set(expand_roles("hr"))
        assert "hr" in result
        assert "employee" in result
        assert "finance" not in result

    def test_management_sees_hr_and_finance(self):
        result = set(expand_roles("management"))
        assert "management" in result
        assert "hr" in result
        assert "finance" in result
        assert "employee" in result
        assert "engineering" not in result   # engineering is not under management

    def test_admin_sees_all_roles(self):
        result = set(expand_roles("admin"))
        assert result == ALL_ROLES

    def test_engineering_sees_employee_and_engineering(self):
        result = set(expand_roles("engineering"))
        assert "engineering" in result
        assert "employee" in result
        assert "hr" not in result

    def test_unknown_role_defaults_to_employee(self):
        result = set(expand_roles("unknown_role_xyz"))
        assert result == {"employee"}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Folder permissions tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFolderPermissions:
    def test_general_folder_is_employee_accessible(self):
        roles = get_allowed_roles_for_path("general")
        assert "employee" in roles

    def test_hr_folder_not_employee_accessible(self):
        roles = get_allowed_roles_for_path("hr")
        assert "employee" not in roles
        assert "hr" in roles

    def test_finance_folder_not_engineering_accessible(self):
        roles = get_allowed_roles_for_path("finance")
        assert "engineering" not in roles
        assert "finance" in roles

    def test_executive_folder_only_management_and_admin(self):
        roles = get_allowed_roles_for_path("executive")
        assert set(roles) == {"management", "admin"}

    def test_unknown_folder_defaults_to_employee(self):
        roles = get_allowed_roles_for_path("some_unknown_dept")
        assert roles == ["employee"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Access check logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanAccess:
    def test_employee_can_access_general_chunk(self):
        assert can_access("employee", ["employee"]) is True

    def test_employee_cannot_access_hr_chunk(self):
        assert can_access("employee", ["hr", "management", "admin"]) is False

    def test_hr_can_access_hr_chunk(self):
        assert can_access("hr", ["hr", "management", "admin"]) is True

    def test_management_can_access_hr_chunk(self):
        # management expands to include "hr"
        assert can_access("management", ["hr", "management", "admin"]) is True

    def test_management_cannot_access_engineering_chunk(self):
        assert can_access("management", ["employee", "engineering"]) is False

    def test_admin_can_access_any_chunk(self):
        for folder, roles in FOLDER_PERMISSIONS.items():
            assert can_access("admin", roles) is True, f"Admin blocked by {folder}"

    def test_engineering_cannot_access_finance_chunk(self):
        assert can_access("engineering", ["finance", "management", "admin"]) is False

    def test_engineering_can_access_general_chunk(self):
        assert can_access("engineering", ["employee"]) is True


# ═══════════════════════════════════════════════════════════════════════════════
# 4. JWT tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestJWT:
    def test_authenticate_valid_user(self):
        user = authenticate_user("alice", "secret")
        assert user is not None
        assert user.role == "hr"

    def test_authenticate_wrong_password(self):
        assert authenticate_user("alice", "wrongpassword") is None

    def test_authenticate_unknown_user(self):
        assert authenticate_user("nobody", "secret") is None

    def test_token_roundtrip(self):
        token = create_access_token("bob", "engineering", "Bob Kumar")
        payload = decode_token(token)
        assert payload["sub"] == "bob"
        assert payload["role"] == "engineering"

    def test_invalid_token_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            decode_token("not.a.valid.jwt")
        assert exc_info.value.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# 5. API endpoint enforcement tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_engine_mock(answer: str = "Test answer.") -> MagicMock:
    node = MagicMock()
    node.node.text = "Context text."
    node.node.metadata = {
        "source": "leave_policy.md",
        "department": "hr",
        "allowed_roles": ["hr", "management", "admin"],
        "file_type": "md",
    }
    node.score = 0.9
    response = MagicMock()
    response.__str__ = lambda self: answer
    response.source_nodes = [node]
    return MagicMock(query=MagicMock(return_value=response))


@pytest.fixture(scope="module")
def api_client():
    """Build a TestClient with auth enabled and a mocked engine."""
    mock_engine = _make_engine_mock()

    with patch("api.main.get_or_build_index", return_value=MagicMock()), \
         patch("api.main.set_index"), \
         patch("api.main.get_index", return_value=MagicMock()), \
         patch("api.main.build_query_engine_for_user", return_value=mock_engine), \
         patch("api.main._build_engine_for", return_value=mock_engine):
        from api.main import app
        with TestClient(app) as c:
            yield c


def _get_token(client: TestClient, username: str) -> str:
    r = client.post("/auth/token", data={"username": username, "password": "secret"})
    assert r.status_code == 200, f"Login failed for {username}: {r.text}"
    return r.json()["access_token"]


class TestAuthEndpoints:
    def test_token_endpoint_returns_jwt(self, api_client):
        r = api_client.post("/auth/token",
                            data={"username": "alice", "password": "secret"})
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "hr"

    def test_token_wrong_password_401(self, api_client):
        r = api_client.post("/auth/token",
                            data={"username": "alice", "password": "wrong"})
        assert r.status_code == 401

    def test_me_returns_user_info(self, api_client):
        token = _get_token(api_client, "alice")
        r = api_client.get("/auth/me",
                           headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["role"] == "hr"

    def test_my_roles_returns_expanded_roles(self, api_client):
        token = _get_token(api_client, "dave")   # management
        r = api_client.get("/auth/my-roles",
                           headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        accessible = r.json()["accessible_roles"]
        assert "hr" in accessible
        assert "finance" in accessible
        assert "management" in accessible

    def test_unauthenticated_query_rejected(self, api_client):
        r = api_client.post("/query", json={"question": "What is our policy?"})
        assert r.status_code == 401

    def test_authenticated_query_succeeds(self, api_client):
        token = _get_token(api_client, "alice")
        r = api_client.post(
            "/query",
            json={"question": "What is the parental leave policy?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert data["user_role"] == "hr"

    def test_query_response_includes_user_role(self, api_client):
        token = _get_token(api_client, "bob")   # engineering
        r = api_client.post(
            "/query",
            json={"question": "How do I set up the dev environment?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["user_role"] == "engineering"

    def test_ingest_requires_admin(self, api_client):
        token = _get_token(api_client, "alice")   # hr, not admin
        r = api_client.post(
            "/ingest",
            json={"force_rebuild": False},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 403

    def test_ingest_allowed_for_admin(self, api_client):
        token = _get_token(api_client, "admin")
        r = api_client.post(
            "/ingest",
            json={"force_rebuild": False},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200

    def test_health_requires_no_auth(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
