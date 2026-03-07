"""
CogniRad++ Authentication Routes  –  MongoDB Edition
Handles user registration, login, profile management, and Google OAuth.
Uses MongoDB (pymongo) for persistence and PyJWT for tokens.
"""

import os
import datetime
import functools
import hashlib
import hmac
import json
import uuid

from flask import Blueprint, request, jsonify

# --- Optional imports ---
try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

try:
    from passlib.hash import pbkdf2_sha256
except ImportError:
    pbkdf2_sha256 = None

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

JWT_SECRET = os.environ.get("JWT_SECRET", "cognirad-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXP_HOURS = int(os.environ.get("JWT_EXP_HOURS", "72"))

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# ---------------------------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------------------------
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "cognirad")

_client = MongoClient(MONGO_URI)
db = _client[MONGO_DB_NAME]
users_collection = db["users"]

# Create unique index on email (idempotent – safe to call on every start)
users_collection.create_index("email", unique=True)

print(f"[Auth] Connected to MongoDB: {MONGO_URI}  db={MONGO_DB_NAME}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    if pbkdf2_sha256:
        return pbkdf2_sha256.hash(password)
    # Fallback (NOT production‑grade – install passlib!)
    salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return f"sha256${salt}${h}"


def _verify_password(password: str, hashed: str) -> bool:
    if not hashed:
        return False
    if pbkdf2_sha256 and hashed.startswith("$pbkdf2"):
        return pbkdf2_sha256.verify(password, hashed)
    parts = hashed.split("$")
    if len(parts) == 3 and parts[0] == "sha256":
        salt, stored = parts[1], parts[2]
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
        return hmac.compare_digest(h, stored)
    return False


def _make_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXP_HOURS),
    }
    if pyjwt:
        return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    import base64
    return base64.urlsafe_b64encode(json.dumps({"sub": user_id}).encode()).decode()


def _decode_token(token: str):
    if pyjwt:
        return pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    import base64
    return json.loads(base64.urlsafe_b64decode(token).decode())


def _user_dict(doc):
    """
    Convert a MongoDB document to a JSON‑safe dict, stripping out
    sensitive fields and converting ObjectId / _id.
    """
    if doc is None:
        return None
    d = dict(doc)
    # Ensure a string "id" field
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    # If we stored our own "id" string, keep it
    d.pop("password_hash", None)
    return d


def _find_user_by_email(email: str):
    return users_collection.find_one({"email": email})


def _find_user_by_id(uid: str):
    return users_collection.find_one({"id": uid})


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

def login_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = _decode_token(token)
            user_doc = _find_user_by_id(payload["sub"])
            if not user_doc:
                return jsonify({"error": "User not found"}), 401
            request.current_user = _user_dict(user_doc)
            request.current_user_id = payload["sub"]
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    role = data.get("role", "radiologist")

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    if _find_user_by_email(email):
        return jsonify({"error": "Email already registered"}), 409

    now = datetime.datetime.utcnow().isoformat()
    user_id = str(uuid.uuid4())

    user_doc = {
        "id": user_id,
        "name": name,
        "email": email,
        "password_hash": _hash_password(password),
        "role": role,
        "avatar_url": "",
        "bio": "",
        "phone": "",
        "department": "",
        "specialization": "",
        "google_id": "",
        "created_at": now,
        "updated_at": now,
    }
    users_collection.insert_one(user_doc)

    token = _make_token(user_id)
    return jsonify({"token": token, "user": _user_dict(_find_user_by_id(user_id))}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    doc = _find_user_by_email(email)
    if not doc or not _verify_password(password, doc.get("password_hash", "")):
        return jsonify({"error": "Invalid email or password"}), 401

    token = _make_token(doc["id"])
    return jsonify({"token": token, "user": _user_dict(doc)})


@auth_bp.route("/google", methods=["POST"])
def google_auth():
    """
    Accepts { token: <Google credential JWT> }.
    Verifies the Google ID token and creates or returns the user.
    """
    data = request.get_json(force=True)
    credential = data.get("token")
    if not credential:
        return jsonify({"error": "Google credential required"}), 400

    # Try google‑auth library first
    google_payload = None
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests
        google_payload = google_id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except Exception:
        # Fallback – decode without verification (dev only)
        try:
            import base64
            parts = credential.split(".")
            if len(parts) >= 2:
                padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
                google_payload = json.loads(base64.urlsafe_b64decode(padded))
        except Exception:
            return jsonify({"error": "Could not verify Google token"}), 401

    if not google_payload:
        return jsonify({"error": "Invalid Google token"}), 401

    g_email = google_payload.get("email", "").lower()
    g_name = google_payload.get("name", g_email.split("@")[0])
    g_picture = google_payload.get("picture", "")
    g_sub = google_payload.get("sub", "")

    doc = _find_user_by_email(g_email)
    if doc:
        user_id = doc["id"]
        if not doc.get("google_id"):
            users_collection.update_one(
                {"id": user_id},
                {"$set": {"google_id": g_sub, "updated_at": datetime.datetime.utcnow().isoformat()}},
            )
    else:
        user_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        users_collection.insert_one({
            "id": user_id,
            "name": g_name,
            "email": g_email,
            "password_hash": "",
            "role": "radiologist",
            "avatar_url": g_picture,
            "bio": "",
            "phone": "",
            "department": "",
            "specialization": "",
            "google_id": g_sub,
            "created_at": now,
            "updated_at": now,
        })

    token = _make_token(user_id)
    return jsonify({"token": token, "user": _user_dict(_find_user_by_id(user_id))})


@auth_bp.route("/profile", methods=["GET"])
@login_required
def get_profile():
    return jsonify({"user": request.current_user})


@auth_bp.route("/profile", methods=["PUT"])
@login_required
def update_profile():
    data = request.get_json(force=True)
    allowed = ["name", "bio", "phone", "department", "specialization", "avatar_url", "role"]
    updates = {k: v for k, v in data.items() if k in allowed and v is not None}

    if not updates:
        return jsonify({"error": "No valid fields to update"}), 400

    updates["updated_at"] = datetime.datetime.utcnow().isoformat()

    users_collection.update_one(
        {"id": request.current_user_id},
        {"$set": updates},
    )

    updated_doc = _find_user_by_id(request.current_user_id)
    return jsonify({"user": _user_dict(updated_doc)})


@auth_bp.route("/change-password", methods=["POST"])
@login_required
def change_password():
    data = request.get_json(force=True)
    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")

    if not new_password or len(new_password) < 6:
        return jsonify({"error": "New password must be at least 6 characters"}), 400

    doc = users_collection.find_one({"id": request.current_user_id})

    if doc.get("password_hash") and not _verify_password(current_password, doc["password_hash"]):
        return jsonify({"error": "Current password is incorrect"}), 401

    users_collection.update_one(
        {"id": request.current_user_id},
        {"$set": {
            "password_hash": _hash_password(new_password),
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }},
    )

    return jsonify({"message": "Password changed successfully"})
