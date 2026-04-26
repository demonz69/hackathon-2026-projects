# MediClaim AI 🏥

> AI-powered medical bill analyzer — know exactly what you're paying for.

## Problem

Medical billing errors affect **1 in 5 patients**. Bills are intentionally opaque. Patients overpay **billions annually** due to errors, duplicates, and vague charges.

## Solution

Upload your bill → **plain-English breakdown in seconds**. Every overcharge flagged, every vague item explained, full FHIR output for interoperability.

## Track

**Health Data & Interoperability** — CareDevi AI Innovation Hackathon 2026

## Architecture

```
PDF Upload → PyMuPDF Extraction → Medical NER Entity Extraction
→ RxNorm + OpenFDA + NLM Enrichment (all free, no key)
→ Llama3-OpenBioLLM-70B Analysis (HuggingFace)
→ CMS Rate Benchmarking (deterministic, no LLM)
→ Severity Classification (HIGH / MEDIUM / LOW)
→ FHIR ExplanationOfBenefit Output
→ React Dashboard
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.11 |
| PDF Parsing | PyMuPDF + Tesseract OCR |
| NER | Medical entity extraction (regex + HuggingFace + Google Gemini) |
| LLM | Llama3-OpenBioLLM-70B via HuggingFace Inference API |
| Enrichment | OpenFDA, RxNorm, NLM Clinical Tables (free, no key) |
| Benchmarks | CMS Medicare Physician Fee Schedule |
| FHIR | Custom ExplanationOfBenefit R4 resource |
| Frontend | React 18, Tailwind CSS, Recharts |

## Key Design Decisions

- **LLM for language only** — never for numeric decisions. All overcharge detection and severity classification is pure deterministic Python against CMS benchmarks.
- **Hybrid NER** — robust regex-based extraction that always works, with optional HuggingFace model enhancement and Google Gemini API fallback.
- **Three free government APIs** — RxNorm (drug normalization), OpenFDA (drug verification + recalls), NLM Clinical Tables (CPT validation). No API keys required.
- **FHIR R4 output** — valid ExplanationOfBenefit resource that plugs into any hospital system, insurance portal, or government health API.

## Data Sources

- **CMS Medicare Physician Fee Schedule** — public domain US government data
- **Synthea** — synthetic patient bills for testing (zero PHI)
- **OpenFDA** — public drug database
- **Insurance coverage** — simulated rules for demo purposes

## Limitations

- OCR may degrade on handwritten or poor-quality scanned bills
- Insurance coverage is simulated, not connected to real insurer data
- CMS rates reflect Medicare; private insurance rates may differ
- Not a substitute for professional billing, legal, or medical advice

## Setup

```bash
# Backend
cd src
pip install -r requirements.txt
cp .env.example .env
# Add your HuggingFace token and GEMINI_API_KEY to .env
cd ..
python -m uvicorn backend.main:app --reload

# Frontend
cd src/frontend
npm install
npm run dev
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/bills/analyze` | Analyze a medical bill PDF |
| GET | `/api/insurance/providers` | List insurance providers |
| GET | `/api/insurance/coverage` | Calculate coverage estimate |
| POST | `/api/fhir/eob` | Generate FHIR ExplanationOfBenefit |
| GET | `/api/fhir/eob/{id}/download` | Download FHIR EOB as JSON |

## Team

Solo participant — @demonz69
