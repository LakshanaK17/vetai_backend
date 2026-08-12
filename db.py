"""Supabase persistence for VetAI (optional).

If SUPABASE_URL / SUPABASE_KEY are not set, every function degrades gracefully to a
no-op so the API still runs locally without a database. The `supabase` package is
imported lazily so the backend works even when it isn't installed.
"""
import os

_client = None
_checked = False


def _get_client():
    """Return a cached Supabase client, or None if not configured/available."""
    global _client, _checked
    if _checked:
        return _client
    _checked = True
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[db] SUPABASE_URL/SUPABASE_KEY not set — persistence disabled.")
        return None
    try:
        from supabase import create_client  # lazy import
        _client = create_client(url, key)
        print("[db] Supabase client ready.")
    except Exception as e:
        print("[db] could not init Supabase client:", e)
        _client = None
    return _client



def save_diagnosis(resp: dict, user_email: str = None):
    c = _get_client()
    if c is None:
        return None
    row = {
        "breed": resp.get("breed"),
        "breed_confidence": resp.get("breedConfidence"),
        "lesion": resp.get("lesion"),
        "lesion_confidence": resp.get("lesionConfidence"),
        "lesion_category": resp.get("lesionCategory"),
        "low_confidence": resp.get("lowConfidence"),
        "treatment": resp.get("treatment"),          
        "diet": resp.get("diet"),                    
        "ai_recommendation": resp.get("aiRecommendation"),
        "image_url": resp.get("image_url"), 
        "lesion_image_url": resp.get("lesion_image_url"),
        "user_email": user_email 
    }
    try:
        res = c.table("diagnoses").insert(row).execute()
        return res.data[0]["id"]
    except Exception as e:
        print("[db] save_diagnosis failed:", e)
        return None


def get_history(email: str):
    c = _get_client()
    if c is None:
        return []
    try:
        
        res = c.table("diagnoses").select("*").eq("user_email", email).order("created_at", desc=True).execute()
        return res.data
    except Exception as e:
        print("[db] get_history failed:", e)
        return []
def signup_user(email: str, password: str, phone: str = None):
    """Sign up a new user via Supabase Auth."""
    c = _get_client()
    if c is None:
        raise Exception("Database not configured. Cannot sign up.")
    
    try:
        # Construct the auth payload
        auth_data = {"email": email, "password": password}
        if phone:
            auth_data["options"] = {"data": {"phone": phone}}
            
        res = c.auth.sign_up(auth_data)
        return res
    except Exception as e:
        print("[db] signup failed:", e)
        raise e

def login_user(email: str, password: str):
    """Log in a user via Supabase Auth."""
    c = _get_client()
    if c is None:
        raise Exception("Database not configured. Cannot log in.")
    
    try:
        res = c.auth.sign_in_with_password({"email": email, "password": password})
        return res
    except Exception as e:
        print("[db] login failed:", e)
        raise e
    
def upload_image(image_bytes: bytes, filename: str):
    """Uploads an image to Supabase storage and returns its public URL."""
    c = _get_client()
    if c is None:
        return None
    try:
        # Uploads to the public 'images' bucket
        c.storage.from_("images").upload(
            file=image_bytes,
            path=filename,
            file_options={"content-type": "image/jpeg"}
        )
        # Retrieve the public URL
        return c.storage.from_("images").get_public_url(filename)
    except Exception as e:
        print("[db] upload_image failed:", e)
        return None