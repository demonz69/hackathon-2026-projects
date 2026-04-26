# Responsible AI — MediClaim AI

## Model Choices & Rationale

- **Llama3-OpenBioLLM-70B**: Medical-domain fine-tuned LLM that reduces hallucination of clinical terminology compared to general-purpose models.
- **Medical NER (regex + HuggingFace + Google Gemini)**: Deterministic entity extraction used where LLM guessing is unacceptable — CPT codes, dollar amounts, and drug names are extracted with pattern matching, not generative AI.
- **LLMs used ONLY for explanation and reasoning**, never for numeric decisions. All overcharge calculations, severity ratings, and benchmark comparisons are pure deterministic Python.

## Data Sources

- **CMS Medicare Fee Schedule**: Public US government data, freely available, used as pricing benchmarks.
- **No real patient data stored**: Bills are processed in memory and immediately discarded. No PHI is persisted to disk or transmitted to third parties beyond the HuggingFace/Google inference APIs.
- **Synthea used for testing**: All demo data is synthetic by design, containing zero Protected Health Information.

## Bias Considerations

- CMS rates reflect **Medicare demographics** — they may not represent pediatric, rural, or uninsured patient populations accurately.
- Insurance simulation is **simplified** and does not account for plan-specific exclusions, prior authorizations, or network restrictions.
- NER extraction may **underperform on non-English** bills or handwritten documents.
- The system has been tested primarily with US medical billing formats.

## Failure Cases & Mitigations

| Failure Case | Mitigation |
|-------------|-----------|
| Bad OCR on scanned bills | Always show raw extracted text alongside AI output so users can verify |
| Hallucinated CPT codes | Cross-validated against NLM Clinical Tables API and CMS benchmark database |
| Overconfident coverage estimates | Labeled "estimated — verify with your insurer" on every output |
| HuggingFace API unavailable | Fallback to Google Gemini API or template-based analysis |
| Unrecognized charges | Flagged as HIGH severity with recommendation to request itemized explanation |

## What This Is NOT

- ❌ Not a legal or financial advisor
- ❌ Not connected to real insurance systems or claims databases
- ❌ Not FDA-cleared as a medical device
- ❌ Not a replacement for professional medical billing review
- ❌ Not designed for production clinical use

**Disclaimer shown on every output screen**: "This analysis is for informational purposes only and does not constitute medical, legal, or financial advice. Always verify charges directly with your healthcare provider and insurance company."

## Privacy & Security

- No user data is stored persistently
- PDF files are processed in-memory and discarded after analysis
- External API calls (HuggingFace, Google) use HTTPS and transmit only anonymized medical text
- No analytics or tracking is implemented
- The application can be run fully locally with model fallbacks

## Transparency

- All severity classifications follow documented, deterministic rules
- CMS benchmark source data is included in the repository for inspection
- The system clearly distinguishes between LLM-generated explanations and deterministic calculations
- Confidence scores are provided for entity extraction results
