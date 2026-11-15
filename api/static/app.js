/**
 * Algions AI Labs - Recommendation Engine Interactive Frontend Client
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const personaSelect = document.getElementById("personaSelect");
  const customUserGroup = document.getElementById("customUserGroup");
  const customUserId = document.getElementById("customUserId");
  const modelSelect = document.getElementById("modelSelect");
  const categoryFilter = document.getElementById("categoryFilter");
  const topKRange = document.getElementById("topKRange");
  const topKDisplay = document.getElementById("topKDisplay");
  const filterHistoryCheck = document.getElementById("filterHistoryCheck");
  const btnRecommend = document.getElementById("btnRecommend");

  const recsGrid = document.getElementById("recsGrid");
  const recsLoading = document.getElementById("recsLoading");
  const activeModelTag = document.getElementById("activeModelTag");
  const latencyPill = document.getElementById("latencyPill");
  const latencyVal = document.getElementById("latencyVal");

  // Profile elements
  const userSegmentBadge = document.getElementById("userSegmentBadge");
  const profilePrimaryCat = document.getElementById("profilePrimaryCat");
  const profilePriceTier = document.getElementById("profilePriceTier");
  const profileHistoryCount = document.getElementById("profileHistoryCount");
  const userRecentHistoryList = document.getElementById("userRecentHistoryList");

  // Tabs
  const tabRecs = document.getElementById("tabRecs");
  const tabBenchmark = document.getElementById("tabBenchmark");
  const viewRecs = document.getElementById("viewRecs");
  const viewBenchmark = document.getElementById("viewBenchmark");
  const metricsTableBody = document.getElementById("metricsTableBody");

  // Modal
  const similarModal = document.getElementById("similarModal");
  const btnCloseModal = document.getElementById("btnCloseModal");
  const modalAnchorTitle = document.getElementById("modalAnchorTitle");
  const similarProductsList = document.getElementById("similarProductsList");

  // Top-K slider update
  topKRange.addEventListener("input", (e) => {
    topKDisplay.textContent = e.target.value;
  });

  // Persona select toggle
  personaSelect.addEventListener("change", (e) => {
    if (e.target.value === "custom") {
      customUserGroup.style.display = "block";
    } else {
      customUserGroup.style.display = "none";
      loadUserProfile(e.target.value);
    }
  });

  customUserId.addEventListener("change", (e) => {
    const val = e.target.value.trim();
    if (val) {
      loadUserProfile(val);
    }
  });

  // Tab switching
  tabRecs.addEventListener("click", () => {
    tabRecs.classList.add("active");
    tabBenchmark.classList.remove("active");
    viewRecs.style.display = "block";
    viewBenchmark.style.display = "none";
  });

  tabBenchmark.addEventListener("click", () => {
    tabBenchmark.classList.add("active");
    tabRecs.classList.remove("active");
    viewRecs.style.display = "none";
    viewBenchmark.style.display = "block";
    loadBenchmarkMetrics();
  });

  // Modal close
  btnCloseModal.addEventListener("click", () => {
    similarModal.style.display = "none";
  });
  window.addEventListener("click", (e) => {
    if (e.target === similarModal) {
      similarModal.style.display = "none";
    }
  });

  // Fetch and display user profile
  async function loadUserProfile(userId) {
    if (userId === "COLD_START_USER") {
      userSegmentBadge.textContent = "New Cold-Start User";
      userSegmentBadge.className = "val badge conf-low";
      profilePrimaryCat.textContent = "None (Unassigned)";
      profilePriceTier.textContent = "Neutral";
      profileHistoryCount.textContent = "0";
      userRecentHistoryList.innerHTML = `<li class="empty-hint">Zero historical interactions. Demonstrates cold-start fallback behavior.</li>`;
      return;
    }

    try {
      const resp = await fetch(`/api/user_profile/${userId}`);
      if (!resp.ok) throw new Error("User profile not found");
      const data = await resp.json();

      userSegmentBadge.textContent = data.is_cold_start ? "Cold-Start User" : "Active Cohort User";
      userSegmentBadge.className = data.is_cold_start ? "val badge conf-low" : "val badge";
      profilePrimaryCat.textContent = data.demographics.primary_category || "Unassigned";
      profilePriceTier.textContent = data.demographics.price_tier || "Standard";
      profileHistoryCount.textContent = data.interaction_count;

      if (data.recent_interactions && data.recent_interactions.length > 0) {
        userRecentHistoryList.innerHTML = data.recent_interactions.map(item => `
          <li class="history-item" title="${item.title}">
            <span class="history-event">${item.event_type}</span> ${item.title}
          </li>
        `).join("");
      } else {
        userRecentHistoryList.innerHTML = `<li class="empty-hint">No prior interactions in training dataset.</li>`;
      }
    } catch (err) {
      console.warn("Could not load user profile:", err);
    }
  }

  // Fetch recommendations
  async function fetchRecommendations() {
    let userId = personaSelect.value;
    if (userId === "custom") {
      userId = customUserId.value.trim() || "USER_0001";
    }

    const payload = {
      user_id: userId,
      top_k: parseInt(topKRange.value, 10),
      category: categoryFilter.value || null,
      filter_history: filterHistoryCheck.checked,
      model_type: modelSelect.value
    };

    recsGrid.innerHTML = "";
    recsLoading.style.display = "block";
    activeModelTag.textContent = modelSelect.options[modelSelect.selectedIndex].text;

    try {
      const resp = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const errorData = await resp.json();
        throw new Error(errorData.detail || "Inference failed");
      }

      const data = await resp.json();
      recsLoading.style.display = "none";

      // Latency pill
      latencyVal.textContent = data.latency_ms;
      latencyPill.style.display = "inline-block";

      renderProductCards(data.recommendations);
    } catch (err) {
      recsLoading.style.display = "none";
      recsGrid.innerHTML = `
        <div style="grid-column: 1/-1; padding: 30px; background: #fee2e2; border-radius: 8px; color: #991b1b; text-align: center;">
          <h3>Inference Error</h3>
          <p>${err.message}</p>
          <p style="margin-top: 8px; font-size: 0.85rem;">Tip: Please ensure models are trained by running <code>python scripts/train.py</code>.</p>
        </div>
      `;
    }
  }

  // Render product cards
  function renderProductCards(items) {
    if (!items || items.length === 0) {
      recsGrid.innerHTML = `
        <div style="grid-column: 1/-1; padding: 40px; text-align: center; color: #64748b;">
          <h3>No matching items found</h3>
          <p>Try widening the category scope or adjusting history filtering.</p>
        </div>
      `;
      return;
    }

    recsGrid.innerHTML = items.map(item => {
      let confClass = "conf-med";
      if (item.confidence_level.includes("High")) confClass = "conf-high";
      if (item.confidence_level.includes("Low")) confClass = "conf-low";

      return `
        <div class="product-card">
          <div>
            <div class="card-top">
              <span class="rank-badge">#${item.rank}</span>
              <span class="cat-pill">${item.category} &bull; ${item.sub_category}</span>
            </div>
            <h3 class="product-title">${item.title}</h3>
            <div class="product-brand">Brand: <strong>${item.brand}</strong></div>
            <div class="product-pricing">
              <span class="product-price">$${item.price.toFixed(2)}</span>
              <span class="score-tag">Score: ${item.score.toFixed(3)}</span>
            </div>
            
            <div class="card-explanation">
              <div class="explanation-header">
                <span>Rationale:</span>
                <span class="confidence-tag ${confClass}">${item.confidence_level}</span>
              </div>
              <p>${item.explanation}</p>
            </div>
          </div>

          <div class="card-actions">
            <button class="btn-small" onclick="window.findSimilar('${item.item_id}', '${item.title.replace(/'/g, "\\'")}')">
              🔍 Find Similar
            </button>
          </div>
        </div>
      `;
    }).join("");
  }

  // Find semantically similar items
  window.findSimilar = async function(itemId, title) {
    modalAnchorTitle.textContent = `Similar to: ${title}`;
    similarProductsList.innerHTML = `<div class="mini-spinner"></div> Loading similar products...`;
    similarModal.style.display = "flex";

    try {
      const resp = await fetch("/api/recommend_similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item_id: itemId, top_k: 5 })
      });

      if (!resp.ok) throw new Error("Could not fetch similar items");
      const data = await resp.json();

      if (data.similar_items && data.similar_items.length > 0) {
        similarProductsList.innerHTML = data.similar_items.map(it => `
          <div class="similar-card">
            <div>
              <strong>${it.title}</strong>
              <div style="font-size: 0.78rem; color: #64748b;">${it.category} &bull; ${it.brand} &bull; $${it.price.toFixed(2)}</div>
            </div>
            <div>
              <span class="metric-pill">Similarity: ${(it.similarity_score * 100).toFixed(1)}%</span>
            </div>
          </div>
        `).join("");
      } else {
        similarProductsList.innerHTML = `<p>No similar items found in the current feature space.</p>`;
      }
    } catch (err) {
      similarProductsList.innerHTML = `<p style="color: #ef4444;">Error: ${err.message}</p>`;
    }
  };

  // Load benchmark metrics into table
  async function loadBenchmarkMetrics() {
    try {
      const resp = await fetch("/api/metrics");
      if (!resp.ok) throw new Error("Could not fetch metrics");
      const data = await resp.json();

      const models = Object.keys(data);
      if (models.length === 0) {
        metricsTableBody.innerHTML = `<tr><td colspan="8">No benchmark metrics found. Run <code>python scripts/evaluate.py</code>.</td></tr>`;
        return;
      }

      metricsTableBody.innerHTML = models.map(m => {
        const row = data[m];
        const isBest = m.toLowerCase().includes("hybrid");
        return `
          <tr ${isBest ? 'style="font-weight: 700; background: #f0fdf4;"' : ''}>
            <td>${m} ${isBest ? '<span class="status-indicator" style="padding: 2px 6px; font-size: 0.7rem;">TOP PERFORMER</span>' : ''}</td>
            <td>${(row["Precision@5"] || 0).toFixed(4)}</td>
            <td>${(row["Precision@10"] || 0).toFixed(4)}</td>
            <td>${(row["Recall@10"] || 0).toFixed(4)}</td>
            <td>${(row["NDCG@10"] || 0).toFixed(4)}</td>
            <td>${(row["MAP@10"] || 0).toFixed(4)}</td>
            <td>${((row["Catalog_Coverage@10"] || 0) * 100).toFixed(1)}%</td>
            <td>${(row["Cold_Users_NDCG@10"] || 0).toFixed(4)}</td>
          </tr>
        `;
      }).join("");
    } catch (err) {
      metricsTableBody.innerHTML = `<tr><td colspan="8" style="color: #ef4444;">Failed to load metrics: ${err.message}</td></tr>`;
    }
  }

  // Initialize
  btnRecommend.addEventListener("click", fetchRecommendations);
  loadUserProfile("USER_0012");
  fetchRecommendations();
});

