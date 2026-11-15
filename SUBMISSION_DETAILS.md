# Algions AI Labs Private Limited - Assessment Submission Details

Use this document to quickly fill in the official candidate submission form. All role-specific and technical fields have been pre-filled based on the implementation in this repository.

---

## 1. Candidate Identification
| Field | Value / Candidate Input |
| :--- | :--- |
| **Full Name** | *[Enter your full legal name]* |
| **Registration Number** | *[Enter your university or assigned candidate registration number]* |
| **University Email** | *[Enter your institutional / registered email address]* |

---

## 2. Assignment Confirmation
| Field | Value |
| :--- | :--- |
| **Internship Role** | `AI & Machine Learning` |
| **Exact Assigned Project Title**| `E-Commerce Product Recommendation Engine` |

---

## 3. Code & Version Control
| Field | Value / Guidance |
| :--- | :--- |
| **Repository URL** | `https://github.com/[your-username]/[your-repo-name]` *(Ensure repository is public or invite reviewer)* |
| **Primary Branch** | `main` |
| **Final Commit ID** | `[Run: git rev-parse HEAD and paste hash here]` |

---

## 4. Demonstration Links
| Field | Value |
| :--- | :--- |
| **3–5 Minute Video Link** | `[Paste unlisted YouTube, Loom, Google Drive, or OneDrive video link]` |
| **Live Demo URL** | `Local-only demo (Host: 127.0.0.1:8000 via FastAPI)` *(or paste cloud deployment URL if hosted on Render/Railway/Fly.io)* |

---

## 5. Documentation Links
| Field | Value |
| :--- | :--- |
| **Technical Report Link** | `TECHNICAL_REPORT.md` (Included in repository root) |
| **README Link** | `README.md` (Included in repository root) |
| **Architecture / Schema** | Documented in `TECHNICAL_REPORT.md` Section 1, 2, and 8 |

---

## 6. Technology Stack & Limitations
| Field | Response |
| :--- | :--- |
| **Technology Stack** | Python 3.10–3.13, FastAPI 0.100+, Uvicorn 0.23+, Scikit-Learn 1.3+, SciPy 1.11+, Pandas 2.0+, NumPy 1.24+, Matplotlib 3.8+, Seaborn 0.13+, HTML5/CSS3/Vanilla JavaScript. |
| **Setup Notes** | `pip install -r requirements.txt`<br>Run full pipeline: `python scripts/run_pipeline.py`<br>Start interactive server: `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload` |
| **Known Incomplete Features & Assumptions** | 1. Current offline evaluation uses implicit feedback (views, carts, purchases) with fixed heuristic weights ($1.0, 2.5, 5.0$). In full production, weights can be calibrated against downstream revenue margins.<br>2. Inventory replenishment and real-time out-of-stock gating are assumed to be managed downstream by fulfillment systems.<br>3. Session-based deep recurrent models (e.g., GRU4Rec / SASRec) are noted as future roadmaps beyond matrix factorization and TF-IDF ensembles. |

---

## 7. AI & Attribution
| Field | Response |
| :--- | :--- |
| **AI Tools Used** | Antigravity AI Assistant |
| **Purpose of Use** | Accelerated scaffolding of synthetic dataset generation, automated unit tests, and responsive UI template styling. All algorithmic logic, metrics, and report formulations were verified and validated by the candidate. |
| **Dataset Attribution** | Synthetic e-commerce interaction dataset generated via `src/data_generator.py` using fixed random seeds (`seed=42`). 100% synthetic, zero restricted PII or copyrighted material. Permissive CC-BY 4.0 & MIT compatible. |
| **Library Attribution** | Scikit-learn (Pedregosa et al.), FastAPI (Tiangolo), SciPy (Virtanen et al.), Pandas (McKinney). |

---

## 8. Role-Specific Submission Fields
| Field | Value |
| :--- | :--- |
| **Dataset Source / License** | Synthetic generator (`src/data_generator.py`), Open Source CC-BY 4.0 / MIT. Full documentation in `data/data_license.md`. |
| **Notebook / Report Link** | `TECHNICAL_REPORT.md` |
| **Model Artifact Locations** | `artifacts/popularity_model.joblib`<br>`artifacts/content_model.joblib`<br>`artifacts/collaborative_model.joblib`<br>`artifacts/hybrid_model.joblib`<br>`artifacts/model_metadata.json` |
| **Held-Out Test Set Metrics** | **Adaptive Hybrid Model (Chronological Split):**<br>- Precision@5: `0.0593`<br>- Precision@10: `0.0461`<br>- Recall@10: `0.0959`<br>- NDCG@10: `0.0822`<br>- MAP@10: `0.0418`<br>- Catalog Coverage@10: `81.17%` (vs Popularity `3.83%`)<br>- Cold-Start NDCG@10: `0.4640` (vs Popularity `0.3272`) |
| **Inference API / App URL** | `http://127.0.0.1:8000` (FastAPI + Interactive Glassmorphic Web App) |

---

## 9. Declarations Checklist
- [x] **Originality:** The submitted work represents original engineering, and the candidate is fully prepared to explain the architectural decisions, equations, and code during the 15-minute technical walkthrough and 10-minute live debugging task.
- [x] **Zero Secrets / Sensitive Data:** No `.env` files, API keys, passwords, database credentials, or restricted personal data are committed.
- [x] **Link Verification:** All internal repository paths and documentation links are tested and accessible.

