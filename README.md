# FIREWISE AI — Complete Full-Stack Project

A React/Vite frontend styled to closely match the supplied FIREWISE AI reference screenshot, plus a FastAPI backend for research document indexing, PDF upload, retrieval-grounded Q&A, comparison and research themes.

## Included features
- Screenshot-inspired dark navy interface, cyan accents, left navigation, hero, feature tiles and status strip.
- Home, AI Assistant, Research Library, Experiments, Analysis, Safety Insights and Reports pages.
- Backend health status and document listing.
- PDF upload and text extraction using `pypdf` (chosen to avoid the PyMuPDF DLL import problem on your Windows setup).
- Keyword retrieval across the starter research catalogue and uploaded PDFs.
- Optional Gemini answer generation grounded in retrieved source excerpts.
- Qualitative programme comparison and evidence/scope notes.
- Text report export.

## Requirements
- Python 3.10 or newer (Python 3.13 is a sensible choice).
- Node.js 20 or newer.
- Gemini API key is optional; without it, the assistant returns retrieved source excerpts.

## Windows setup

### 1. Backend — terminal 1
Open PowerShell in the project root:

```powershell
cd backend
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `backend/.env` and add your key if you have one:

```env
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Start backend:

```powershell
uvicorn main:app --reload
```

Check `http://127.0.0.1:8000/api/health` and API docs at `http://127.0.0.1:8000/docs`.

If `py -3.13` is unavailable, use `py -m venv .venv` instead. If PowerShell activation is blocked, use Command Prompt and run `.venv\Scripts\activate.bat`.

### 2. Frontend — terminal 2
Open a second terminal at the project root:

```powershell
cd frontend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npm.cmd install
npm.cmd run dev
```

Using `npm.cmd` avoids PowerShell's `npm.ps1` execution-policy error. Open the local URL Vite prints, normally `http://localhost:5173`.

The frontend defaults to `http://127.0.0.1:8000`. If needed, create `frontend/.env` with `VITE_API_URL=http://127.0.0.1:8000`, then restart Vite.

## Important scientific limitations
- The included catalogue contains qualitative research summaries, not a verified numerical dataset.
- The comparison endpoint is qualitative; it does not compute experimental metrics.
- Do not present AI-generated interpretations as NASA findings. Check original source documents and page references.
- This is a research-support prototype, not a certified spacecraft fire simulator, operational risk predictor, detector, or safety controller.
- Confirm at least one NASA dataset/source from the exact challenge Resources tab is used in the submitted project.

## Project structure
```text
FIREWISE-AI/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── data/catalogue.json
│   └── uploads/
├── frontend/
│   ├── public/engine-test-reference.jpg
│   ├── src/App.jsx
│   ├── src/main.jsx
│   ├── src/styles.css
│   └── package.json
├── documentation/
└── README.md
```
