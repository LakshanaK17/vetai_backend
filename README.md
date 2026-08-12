# VetAI — FastAPI backend (Supabase + Railway)

Serves the VetAI pipeline (breed model → lesion model → BSAVA rule retrieval → LLM
decision layer), stores each diagnosis in **Supabase**, and is built to deploy on
**Railway** for a **Vercel** frontend.

## Files
```
main.py             FastAPI app: /health /breed /diagnose /history
knowledge_base.py   BSAVA rules + breed-aware diet + retrieval
llm_layer.py        grounded LLM generation (USE_LLM=1)
db.py               Supabase persistence (graceful no-op if unset)
supabase_schema.sql table to create in Supabase
requirements.txt    dependencies
Dockerfile          container (listens on $PORT)
railway.toml        Railway build/deploy config
.env.example        configuration template
```

## Endpoints
| Method | Path         | Body (multipart)            | Returns |
|--------|--------------|-----------------------------|---------|
| GET    | `/health`    | –                           | status + config |
| POST   | `/breed`     | `dog_image`                 | `{breed, breedConfidence}` |
| POST   | `/diagnose`  | `dog_image`, `lesion_image` | full result + `id` (saved to Supabase) |
| GET    | `/history?limit=50` | –                    | `{items: [...]}` recent diagnoses |

---

## 1. Supabase (database)
1. Create a project at supabase.com.
2. Dashboard → **SQL Editor** → paste `supabase_schema.sql` → **Run** (creates the `diagnoses` table).
3. Dashboard → **Project Settings → API**, copy:
   - **Project URL** → `SUPABASE_URL`
   - **service_role** key (secret!) → `SUPABASE_KEY`
The backend writes with the service-role key (server-side only). Never put it in the frontend.

## 2. Railway (host the backend)
Railway has no GPU, so the LLM runs on CPU — it works but is slow and needs RAM. Two options:
- keep `LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct` and pick an instance with **≥ 4 GB RAM**, or
- set `LLM_MODEL=google/flan-t5-large` (lighter, faster on CPU), or
- set `USE_LLM=0` to run rule-based only and keep the LLM for a local/GPU run.

**Deploy:**
1. Put the trained models in `./models/` (`breed_image_final.keras`, `lesion_image_final.keras`)
   so they are baked into the image. (EfficientNetB0 files are small, ~20–30 MB each.)
2. Push this folder to a GitHub repo.
3. Railway → **New Project → Deploy from GitHub repo**. It auto-detects the `Dockerfile`.
4. Railway → **Variables**, add:
   ```
   USE_LLM=1
   LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
   SUPABASE_URL=...
   SUPABASE_KEY=...            # service_role
   ALLOWED_ORIGINS=https://your-app.vercel.app
   BREED_CLASSES=...           # exact training order (see note below)
   LESION_CLASSES=...
   ```
5. Deploy → Railway gives you a public URL like `https://vetai-api-production.up.railway.app`.
   Test it: open `<url>/docs`.

> Railway injects `$PORT`; the Dockerfile/`railway.toml` already bind to it.

## 3. Vercel (frontend) — make them compatible
1. In the Next.js project set an env var: `NEXT_PUBLIC_API_URL = https://<your-railway-url>`.
2. Redeploy the frontend on Vercel.
3. Back on Railway, set `ALLOWED_ORIGINS` to your exact Vercel domain (e.g.
   `https://vetai.vercel.app`). That's the only thing needed for CORS — the frontend just
   calls the Railway URL.

## 4. Local run
```bash
pip install -r requirements.txt
mkdir models   # copy the two .keras files here
cp .env.example .env   # fill in values
uvicorn main:app --port 8000        # http://localhost:8000/docs
```

## Important: class-name order
`image_dataset_from_directory` orders classes with `sorted()` on the **folder names** — which
is NOT the same as alphabetical by display name. In this dataset the Shih-Tzu / Yorkshire /
Doberman folders are named with `n0…` synset prefixes, so they sort at the "n" position (after
*labrador*); and capitalised lesion folders sort before lowercase ones. The defaults in `main.py`
already use the exact order from your training `classification_report`:

- **Breed:** `Chihuahua, Cocker_Spaniel, French_Bulldog, German_Shepherd, Golden_Retriever, Labrador_Retriever, Shih_Tzu, Yorkshire_Terrier, Doberman, Rottweiler, Siberian_Husky`
- **Lesion:** `Dermatitis, Fungal_infections, Healthy, Hypersensitivity, Demodicosis, Ringworm`

A wrong order gives shifted labels (e.g. Golden predicted as German).
