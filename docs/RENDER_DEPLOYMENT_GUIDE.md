# CircuitGuard AI: Render Deployment Guide (100% Free Tier)

This guide walks you through deploying **CircuitGuard AI** on [Render.com](https://render.com) as a single, production-ready web service.

---

## ⚡ Architecture on Render

* **Single Web Service:** The FastAPI backend serves both the REST API endpoints (`/api/...`) and the built React Single-Page Application (`/` and `/assets/...`) on the same port.
* **Zero CORS Configuration:** Because frontend and backend are hosted on the same origin, API requests never fail due to cross-origin security restrictions.
* **Cost:** **$0 / month (Render Free Tier)**.

---

## Step 1: Push Code to GitHub

Open terminal in the project directory:

```bash
# Initialize git if not already done
git init

# Add all files (models, backend, frontend, dataset)
git add .

# Commit
git commit -m "Deploy CircuitGuard AI with 4-model suite and React dashboard"

# Link to your GitHub repository and push
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
git branch -M main
git push -u origin main
```

---

## Step 2: Deploy on Render

### Option A: Using Render Blueprint (Automatic — 1 Click)

Because this repository includes a [`render.yaml`](../render.yaml) file:

1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub repository.
4. Render will automatically read `render.yaml` and configure everything:
   * **Name:** `circuitguard-ai`
   * **Environment:** `Python`
   * **Build Command:** `./build.sh`
   * **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Click **Apply**.

---

### Option B: Manual Setup via Web Service

If you prefer to configure manually:

1. On [Render Dashboard](https://dashboard.render.com), click **New +** → **Web Service**.
2. Connect your GitHub repository.
3. Configure the following fields:
   * **Name:** `circuitguard-ai`
   * **Region:** Any (e.g. *Oregon (US West)* or *Frankfurt (EU)*)
   * **Branch:** `main`
   * **Runtime:** `Python 3`
   * **Build Command:** `./build.sh`
   * **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   * **Instance Type:** `Free`
4. Expand **Advanced** → **Add Environment Variable**:
   * `PYTHON_VERSION` = `3.11.9`
   * `OMP_NUM_THREADS` = `1`
   * `KMP_DUPLICATE_LIB_OK` = `True`
5. Click **Deploy Web Service**.

---

## Step 3: Verification

Once the build finishes (takes ~2 minutes):
1. Render will assign you a live public URL (e.g., `https://circuitguard-ai.onrender.com`).
2. Open the URL in any browser:
   * **Dashboard:** `https://your-service.onrender.com`
   * **Interactive API Docs:** `https://your-service.onrender.com/docs`
   * **Health Check:** `https://your-service.onrender.com/api/health`

---

## 💡 Important Notes for Free Tier

* **Spin-down Behavior:** Free instances on Render spin down after 15 minutes of inactivity. When a new visitor opens the URL, it takes 30–50 seconds to wake up. Once awake, all predictions run in sub-50ms.
* **Memory Footprint:** The 4 models loaded together occupy approximately ~340 MB of RAM, well within Render's 512 MB free tier allowance.

