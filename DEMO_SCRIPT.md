# Algions AI Labs - Video Demonstration Script (3–5 Minutes)

**Assessment:** E-Commerce Product Recommendation Engine | AI & Machine Learning Track  
**Candidate Name:** [Your Name]  
**Target Duration:** 3 minutes 30 seconds – 4 minutes 30 seconds  

---

## Video Recording Structure

```
[0:00 - 0:45] Phase 1: Problem Statement, Framing & Decision-Support Scope
[0:45 - 1:30] Phase 2: Data Synthesis, Leakage-Free Chronological Split & EDA
[1:30 - 2:45] Phase 3: Model Architecture Benchmark & Offline Test Results
[2:45 - 4:00] Phase 4: Live Inference Demonstration (Active User vs Cold-Start)
[4:00 - 4:30] Phase 5: Error Analysis, Failure Modes, and Responsible AI Oversight
```

---

## Detailed Step-by-Step Script & Narration

### Phase 1: Problem Statement & Framing (0:00 – 0:45)
- **Screen:** Show the project repository in your IDE/editor, highlighting `README.md` and the architecture diagram.
- **Narration:**
  > "Hello reviewers. Today I am presenting my solution for the Algions AI Labs E-Commerce Product Recommendation Engine assessment.
  >
  > The business problem is to assist active shoppers in discovering relevant catalog items during a session while giving merchandising teams a decision-support tool that presents clear, explainable signals rather than an opaque black box.
  >
  > Our prediction unit is a user-item pair at a specific session entry timestamp $t_0$. Our target is the implicit engagement affinity score aggregated across views, carts, and purchases. Crucially, this model is designed for human-in-the-loop decision support: it informs catalog discovery while respecting inventory availability and merchandising policies."

---

### Phase 2: Data Synthesis, Leakage Prevention & EDA (0:45 – 1:30)
- **Screen:** Open `artifacts/figures/interaction_distribution.png` and `data/data_license.md`, then show terminal running `python scripts/run_pipeline.py`.
- **Narration:**
  > "For data sourcing, we synthesized a fully reproducible 90-day e-commerce dataset containing 600 products across 6 departments, 2,500 users, and 65,036 interactions using a fixed random seed. As shown in our EDA plots, the item interaction distribution follows an empirical power law where the top 20% of products capture 68% of engagement, mimicking real-world retail catalogs with high matrix sparsity at 96.53%.
  >
  > To strictly prevent future data leakage, we avoided random splits and implemented a chronological cutoff at the 80th percentile timestamp. Training interactions occur strictly prior to October 12, 2026, while testing interactions occur exclusively in the subsequent time window. Our automated unit test mathematically verifies that max train timestamp is strictly less than min test timestamp, ensuring zero future leakage."

---

### Phase 3: Model Benchmark & Test Evaluation (1:30 – 2:45)
- **Screen:** Show the benchmark comparison table in terminal or browser (`artifacts/figures/model_comparison.png`).
- **Narration:**
  > "We developed and benchmarked four distinct approaches on the held-out test set of 1,694 future test users:
  > 1. A recency-decayed Popularity Baseline
  > 2. A Content-Based TF-IDF metadata model
  > 3. A Collaborative Matrix Factorization model using Truncated SVD with 30 latent components
  > 4. Our proposed Adaptive Hybrid Recommender that dynamically blends collaborative and content signals.
  >
  > While the Popularity Baseline achieves high precision on head items, it suffers from severe Catalog Coverage Collapse—recommending only 3.8% of the catalog. 
  >
  > In contrast, our Adaptive Hybrid model achieves 81.17% catalog coverage—a 21-fold improvement—while preserving strong ranking quality at 0.0822 NDCG@10. Furthermore, on cold-start users with zero training history, our Hybrid model achieves 0.4640 NDCG@10, outperforming all standalone models by over 41%."

---

### Phase 4: Live Demonstration through Inference UI (2:45 – 4:00)
- **Screen:** Open the live browser dashboard at `http://127.0.0.1:8000`.
- **Action 1:** Select **User 12 (Tech Enthusiast)** with **Adaptive Hybrid** and click **Generate Recommendations**.
- **Narration:**
  > "Now let's demonstrate real-time inference on our live FastAPI interface. Here we select User 12, an active shopper with extensive electronics history. Notice the inference latency is just 5 milliseconds. The model recommends high-affinity audio and computing products, complete with confidence levels and human-interpretable explanations such as 'Attributes closely match your browsing interest in Audio & Headphones'."
- **Action 2:** Switch persona to **Cold-Start User** and click **Generate Recommendations**.
- **Narration:**
  > "Now watch how the system handles a brand-new user with zero prior interactions. The dynamic routing immediately detects zero history, sets collaborative weight to zero, and falls back to category-conditioned popularity and top-rated items. This avoids runtime errors and delivers instant discovery value."
- **Action 3:** Click **Find Similar** on a product card to show the item-to-item similarity modal.

---

### Phase 5: Error Analysis, Limitations & Ethical Oversight (4:00 – 4:30)
- **Screen:** Show `TECHNICAL_REPORT.md` Section 6 and 7.
- **Narration:**
  > "During error analysis, we observed that false positives predominantly occur during cross-category session transitions—such as a user moving from running shoes to nutrition supplements. We mitigate this through an exploration factor.
  >
  > In terms of responsible AI:
  > - No PII or sensitive demographic data is collected or stored.
  > - Filter bubbles are counteracted by expanding catalog coverage to 81%.
  > - The model strictly functions as a decision-support tool that respects merchandiser inventory overrides.
  >
  > All 18 automated tests pass cleanly. Thank you for your time and consideration."

---

## Tips for Recording
1. **Tooling:** Use OBS Studio, Loom, or Windows Game Bar (`Win + G`).
2. **Resolution:** 1080p full screen. Keep the browser and terminal visible and clear.
3. **Pacing:** Speak clearly and keep transitions smooth.

