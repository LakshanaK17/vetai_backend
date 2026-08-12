"""VetAI FastAPI backend.

Pipeline: breed image -> breed model; lesion image -> lesion model;
(breed, lesion) -> BSAVA rule retrieval + breed-aware diet; optional LLM decision layer.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
Config via environment variables (see .env.example / README.md).
"""
import io
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form

import knowledge_base as kb
import db

# ---------------- configuration ----------------
MODEL_DIR      = os.getenv("MODEL_DIR", "./models")
BREED_MODEL    = os.getenv("BREED_MODEL", "breed_image_final.keras")
LESION_MODEL   = os.getenv("LESION_MODEL", "lesion_image_final.keras")
IMG_SIZE       = (224, 224)
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.50"))
USE_LLM        = os.getenv("USE_LLM", "0") == "1"
LLM_MODEL      = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

# IMPORTANT: these MUST match the alphabetical class order used during training.
# IMPORTANT: this MUST match the exact class_names order your training notebooks printed.
# TF sorts folder names with sorted(): because Shih-Tzu/Yorkshire/Doberman folders are named
# with "n0..." synset prefixes, they sort at the "n" position (after labrador), NOT alphabetically
# by the display name. This order is taken verbatim from the breed classification_report.
BREED_CLASSES = os.getenv("BREED_CLASSES", "").split(",") if os.getenv("BREED_CLASSES") else [
    "Chihuahua", "Cocker_Spaniel", "French_Bulldog", "German_Shepherd", "Golden_Retriever",
    "Labrador_Retriever", "Shih_Tzu", "Yorkshire_Terrier", "Doberman", "Rottweiler", "Siberian_Husky",
]
# Lesion folders: capitalised names sort before lowercase ones -> demodicosis/ringworm come last.
LESION_CLASSES = os.getenv("LESION_CLASSES", "").split(",") if os.getenv("LESION_CLASSES") else [
    "Dermatitis", "Fungal_infections", "Healthy", "Hypersensitivity", "Demodicosis", "Ringworm",
]

# ---------------- model loading ----------------
_breed_model = None
_lesion_model = None

def _load_models():
    global _breed_model, _lesion_model
    import tensorflow as tf
    bpath = os.path.join(MODEL_DIR, BREED_MODEL)
    lpath = os.path.join(MODEL_DIR, LESION_MODEL)
    if not os.path.exists(bpath) or not os.path.exists(lpath):
        raise FileNotFoundError(
            f"Model files not found. Expected:\n  {bpath}\n  {lpath}\n"
            f"Set MODEL_DIR or copy the .keras files there.")
    _breed_model = tf.keras.models.load_model(bpath)
    _lesion_model = tf.keras.models.load_model(lpath)
    print(f"[main] models loaded from {MODEL_DIR}")

def _read_image(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(img, dtype=np.float32), 0)  # EfficientNet preprocess is inside the model

def _predict(model, classes, raw: bytes):
    probs = model.predict(_read_image(raw), verbose=0)[0]
    i = int(np.argmax(probs))
    return classes[i], float(probs[i])

# ---------------- app ----------------
app = FastAPI(title="VetAI API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    _load_models()
    if USE_LLM:
        import llm_layer
        llm_layer.load(LLM_MODEL)
    print(f"[main] ready. USE_LLM={USE_LLM}")

@app.get("/health")
def health():
    return {"status": "ok", "use_llm": USE_LLM,
            "breed_classes": len(BREED_CLASSES), "lesion_classes": len(LESION_CLASSES)}

@app.post("/breed")
async def breed(dog_image: UploadFile = File(...)):
    """Step 1: identify the breed from a full-body photo."""
    label, conf = _predict(_breed_model, BREED_CLASSES, await dog_image.read())
    return {"breed": label, "breedConfidence": round(conf * 100, 1)}

def _build_response(breed_label, breed_conf, lesion_label, lesion_conf):
    rec = kb.retrieve(breed_label, lesion_label)
    r, d = rec["rule"], rec["diet"]
    ai_text = None
    if USE_LLM and rec["category"] != "healthy":
        try:
            import llm_layer
            ai_text = llm_layer.generate(rec)
        except Exception as e:  # never fail the request because of the LLM
            ai_text = None
            print("[main] LLM generation failed:", e)
    return {
        "breed": breed_label, "breedConfidence": round(breed_conf * 100, 1),
        "lesion": lesion_label, "lesionConfidence": round(lesion_conf * 100, 1),
        "lesionCategory": rec["category"],
        "lowConfidence": (breed_conf < CONF_THRESHOLD or lesion_conf < CONF_THRESHOLD),
        "treatment": {
            "recommendation": r["agent"],
            "source": r["source"],
            "ruleTrigger": r["trigger"],
            "exactRuleHit": "Yes" if r["matched"] else "No (generic BSAVA fallback)",
        },
        "diet": {
            "profile": d["profile"],
            "recommended": d["recommended"],
            "quantity": d["quantity"],
            "avoid": d["avoid"],
            "conditionTip": rec["diet_modifier"],
        },
        "aiRecommendation": ai_text,
    }

# @app.post("/diagnose")
# async def diagnose(dog_image: UploadFile = File(...), lesion_image: UploadFile = File(...)):
#     """Full pipeline: breed + lesion + rule-grounded recommendation (+ LLM if enabled)."""
#     breed_label, breed_conf = _predict(_breed_model, BREED_CLASSES, await dog_image.read())
#     lesion_label, lesion_conf = _predict(_lesion_model, LESION_CLASSES, await lesion_image.read())
#     resp = _build_response(breed_label, breed_conf, lesion_label, lesion_conf)
#     resp["id"] = db.save_diagnosis(resp)   # best-effort persistence to Supabase
#     return resp

@app.post("/diagnose")
async def diagnose(
    dog_image: UploadFile = File(...), 
    lesion_image: UploadFile = File(...),
    user_email: str = Form(None) # <--- NEW: Accept email from frontend
):
    dog_bytes = await dog_image.read()
    lesion_bytes = await lesion_image.read()
    
    breed_label, breed_conf = _predict(_breed_model, BREED_CLASSES, dog_bytes)
    lesion_label, lesion_conf = _predict(_lesion_model, LESION_CLASSES, lesion_bytes)
    
    image_url = None
    lesion_image_url = None
    if db._get_client() is not None:
        dog_filename = f"dog_{uuid.uuid4()}.jpg"
        lesion_filename = f"lesion_{uuid.uuid4()}.jpg"
        image_url = db.upload_image(dog_bytes, dog_filename)
        lesion_image_url = db.upload_image(lesion_bytes, lesion_filename)
    
    resp = _build_response(breed_label, breed_conf, lesion_label, lesion_conf)
    resp["image_url"] = image_url
    resp["lesion_image_url"] = lesion_image_url
    
    # NEW: Pass the email to the database
    resp["id"] = db.save_diagnosis(resp, user_email) 
    return resp

# Update your /history endpoint to require the email
@app.get("/history")
async def history(email: str = None): # <--- NEW: Accept email as a query parameter
    if not email:
        return {"items": []}
    
    data = db.get_history(email)
    return {"items": data}

# ---------------- auth models & endpoints ----------------
class SignupRequest(BaseModel):
    email: str
    password: str
    phone: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/signup")
def signup(req: SignupRequest):
    try:
        res = db.signup_user(req.email, req.password, req.phone)
        user_data = res.user.model_dump() if res.user else None
        return {"message": "Signup successful", "user": user_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
def login(req: LoginRequest):
    try:
        res = db.login_user(req.email, req.password)
        user_data = res.user.model_dump() if res.user else None
        return {
            "message": "Login successful", 
            "access_token": res.session.access_token if res.session else None,
            "user": user_data
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid email or password")