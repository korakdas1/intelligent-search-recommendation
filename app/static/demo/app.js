(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const searchForm = $("search-form");
  const retrievalMode = $("retrieval-mode");
  const fusionField = $("fusion-field");
  const hybridOptions = $("hybrid-options");
  const useLtr = $("use-ltr");
  const usePersonalization = $("use-personalization");
  const userField = $("user-field");
  const searchUserId = $("search-user-id");
  const searchSubmit = $("search-submit");
  const searchStatus = $("search-status");

  const SOURCE_BADGES = {
    keyword: ["Keyword"],
    semantic: ["Semantic"],
    hybrid: ["Hybrid"],
    hybrid_ltr: ["Hybrid", "LTR"],
    content_user: ["Content"],
    content_item: ["Similarity"],
    cf: ["Collaborative"],
    popularity: ["Popular"],
  };

  let searchAbort = null;

  function hasValue(value) {
    return value !== null && value !== undefined && String(value).trim() !== "";
  }

  function displayValue(value) {
    if (!hasValue(value)) return "Not available";
    return String(value);
  }

  function isGenericCategory(category) {
    return /^all\s*beauty$/i.test(String(category).trim());
  }

  function formatScore(value) {
    const number = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(number)) return displayValue(value);
    return number.toFixed(4);
  }

  function formatCount(value) {
    const number = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(number)) return displayValue(value);
    return String(Math.round(number));
  }

  function sourceBadges(source) {
    if (!hasValue(source)) return [];
    return SOURCE_BADGES[source] || [String(source)];
  }

  function setHidden(el, hidden) {
    el.hidden = hidden;
  }

  function showError(id, message) {
    const el = $(id);
    el.textContent = message;
    setHidden(el, !message);
  }

  function clearCards(id) {
    const el = $(id);
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function renderMeta(id, chips) {
    const root = $(id);
    root.replaceChildren();
    const visible = chips.filter(Boolean);
    for (const text of visible) {
      root.appendChild(el("span", "meta-chip", text));
    }
    setHidden(root, visible.length === 0);
  }

  function appendDetail(dl, label, value, options) {
    const opts = options || {};
    const dt = el("dt", null, label);
    const dd = el("dd", opts.mono ? "mono" : null, displayValue(value));
    if (opts.title) dd.title = opts.title;
    dl.appendChild(dt);
    dl.appendChild(dd);
  }

  function renderCards(containerId, items, options) {
    const opts = options || {};
    const emptyMessage = opts.emptyMessage || "No matching products found.";
    const showFindSimilar = Boolean(opts.showFindSimilar);
    const popularity = opts.variant === "popular";
    const root = $(containerId);
    clearCards(containerId);
    if (!items || items.length === 0) {
      root.appendChild(el("p", "empty", emptyMessage));
      return;
    }
    for (const item of items) {
      const card = el("article", "card");
      const top = el("div", "card-top");
      if (hasValue(item.rank)) {
        top.appendChild(el("span", "rank-badge", `#${item.rank}`));
      }
      const badges = el("div", "badge-row");
      for (const label of sourceBadges(item.source)) {
        badges.appendChild(el("span", "badge", label));
      }
      top.appendChild(badges);
      card.appendChild(top);

      const title = el("h3", null, displayValue(item.title));
      card.appendChild(title);

      if (hasValue(item.brand)) {
        card.appendChild(el("p", "card-brand", String(item.brand)));
      }
      if (hasValue(item.price)) {
        card.appendChild(el("p", "card-price", String(item.price)));
      }
      if (hasValue(item.category) && !isGenericCategory(item.category)) {
        card.appendChild(el("p", "card-category", String(item.category)));
      }
      if (popularity && hasValue(item.score)) {
        card.appendChild(
          el("p", "card-count", `Historical interactions: ${formatCount(item.score)}`)
        );
      }

      if (showFindSimilar && hasValue(item.product_id)) {
        const actions = el("div", "card-actions");
        const findBtn = el("button", "card-action", "Find similar");
        findBtn.type = "button";
        findBtn.addEventListener("click", () => useForSimilar(item.product_id));
        actions.appendChild(findBtn);
        card.appendChild(actions);
      }

      const details = el("details", "tech");
      const summary = el("summary", null, "Technical details");
      details.appendChild(summary);
      const dl = document.createElement("dl");
      appendDetail(dl, "Rank", item.rank);
      appendDetail(dl, "Product ID", item.product_id, { mono: true });
      if (popularity) {
        appendDetail(dl, "Historical interactions", formatCount(item.score), {
          title: hasValue(item.score) ? String(item.score) : undefined,
        });
      } else {
        appendDetail(dl, "Score", formatScore(item.score), {
          mono: true,
          title: hasValue(item.score) ? String(item.score) : undefined,
        });
      }
      appendDetail(dl, "Source", item.source, { mono: true });
      appendDetail(dl, "Brand", item.brand);
      appendDetail(dl, "Category", item.category);
      appendDetail(dl, "Price", item.price);
      details.appendChild(dl);
      if (hasValue(item.product_id)) {
        const copyBtn = el("button", "card-action", "Copy product ID");
        copyBtn.type = "button";
        copyBtn.addEventListener("click", () => copyProductId(item.product_id, copyBtn));
        details.appendChild(copyBtn);
      }
      card.appendChild(details);
      root.appendChild(card);
    }
  }

  function useForSimilar(productId) {
    $("similar-id").value = productId;
    $("similar-id").focus();
    $("similar-heading").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function copyProductId(productId, button) {
    const previous = button.textContent;
    try {
      await navigator.clipboard.writeText(productId);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Copy failed";
    }
    window.setTimeout(() => {
      button.textContent = previous;
    }, 1200);
  }

  function userMessage(status, payload) {
    if (!payload) return "The service is unavailable.";
    if (typeof payload.detail === "string") return payload.detail;
    if (payload.detail && typeof payload.detail === "object") {
      if (payload.detail.message) return payload.detail.message;
      if (payload.detail.status === "semantic_unavailable") {
        return "Semantic index artifacts are not available.";
      }
      if (payload.detail.status === "ranker_unavailable") {
        return "The learning-to-rank model is not available.";
      }
      if (payload.detail.status === "cf_unavailable") {
        return "The collaborative filtering model is not available.";
      }
    }
    if (Array.isArray(payload.detail) && payload.detail[0] && payload.detail[0].msg) {
      return payload.detail[0].msg;
    }
    if (status === 404) return "That ID was not found.";
    if (status === 422) return "Please check the form inputs.";
    if (status >= 500) return "The service could not complete this request.";
    return "Request failed.";
  }

  async function requestJson(url, options) {
    const started = performance.now();
    const response = await fetch(url, options);
    const elapsed = Math.round(performance.now() - started);
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      const error = new Error(userMessage(response.status, payload));
      error.status = response.status;
      throw error;
    }
    return { payload, elapsed };
  }

  function syncSearchControls() {
    const hybrid = retrievalMode.value === "hybrid";
    fusionField.hidden = !hybrid;
    hybridOptions.hidden = !hybrid;
    if (!hybrid) {
      useLtr.checked = false;
      usePersonalization.checked = false;
    }
    const showUser = hybrid && usePersonalization.checked;
    userField.hidden = !showUser;
    searchUserId.disabled = !showUser;
    if (showUser) searchUserId.focus();
  }

  async function loadReady() {
    const root = $("system-status");
    try {
      const { payload } = await requestJson("/ready");
      const checks = payload.checks || {};
      root.replaceChildren();
      const labels = {
        database: "Database",
        semantic_index: "Semantic index",
        ltr_ranker: "Learning-to-rank",
        cf_model: "Collaborative filtering",
      };
      for (const [key, label] of Object.entries(labels)) {
        const pill = document.createElement("span");
        const value = checks[key] || "unknown";
        pill.className = `pill ${value === "ok" || value === "skipped" ? "ok" : "bad"}`;
        pill.textContent = `${label}: ${value}`;
        root.appendChild(pill);
      }
    } catch {
      root.textContent = "Could not read /ready. The API may not be running.";
    }
  }

  searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("search-error", "");
    const query = $("query").value.trim();
    if (!query) {
      showError("search-error", "Enter a search query.");
      return;
    }
    const mode = retrievalMode.value;
    const body = {
      query,
      top_k: Number($("search-top-k").value),
      retrieval_mode: mode,
      rerank_mode: "none",
      personalization_mode: "none",
    };
    if (mode === "hybrid") {
      body.fusion_method = $("fusion-method").value;
      if (useLtr.checked) body.rerank_mode = "ltr";
      if (usePersonalization.checked) {
        const userId = searchUserId.value.trim();
        if (!userId) {
          showError("search-error", "Personalized reranking needs a user ID.");
          searchUserId.focus();
          return;
        }
        body.personalization_mode = "bounded";
        body.user_id = userId;
      }
    }
    if (searchAbort) searchAbort.abort();
    searchAbort = new AbortController();
    searchSubmit.disabled = true;
    searchStatus.textContent = "Searching…";
    try {
      const { payload, elapsed } = await requestJson("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: searchAbort.signal,
      });
      const chips = [
        payload.retrieval_mode === "keyword" ? "Keyword" : null,
        payload.retrieval_mode === "semantic" ? "Semantic" : null,
        payload.retrieval_mode === "hybrid" ? "Hybrid" : null,
        payload.fusion_method === "weighted" ? "Weighted" : null,
        payload.fusion_method === "rrf" ? "RRF" : null,
        payload.rerank_mode === "ltr" ? "LTR" : null,
        payload.personalization_applied ? "Personalized" : null,
        `${payload.results.length} result${payload.results.length === 1 ? "" : "s"}`,
        `Local request: ${elapsed} ms`,
      ];
      renderMeta("search-meta", chips);
      renderCards("search-results", payload.results, {
        showFindSimilar: true,
        emptyMessage: "No matching products found.",
      });
      searchStatus.textContent = `${payload.results.length} result(s)`;
    } catch (error) {
      if (error.name === "AbortError") return;
      showError("search-error", error.message);
      renderMeta("search-meta", []);
      clearCards("search-results");
      searchStatus.textContent = "";
    } finally {
      searchSubmit.disabled = false;
    }
  });

  $("similar-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("similar-error", "");
    const productId = $("similar-id").value.trim();
    const topK = $("similar-top-k").value;
    const submit = $("similar-submit");
    submit.disabled = true;
    $("similar-status").textContent = "Finding similar…";
    try {
      const { payload, elapsed } = await requestJson(
        `/recommendations/similar/${encodeURIComponent(productId)}?top_k=${topK}`
      );
      renderMeta("similar-meta", ["Similarity", `Local request: ${elapsed} ms`]);
      renderCards("similar-results", payload.results, {
        emptyMessage: "No similar products found.",
      });
      $("similar-status").textContent = `${payload.results.length} result(s)`;
    } catch (error) {
      showError("similar-error", error.message);
      renderMeta("similar-meta", []);
      clearCards("similar-results");
      $("similar-status").textContent = "";
    } finally {
      submit.disabled = false;
    }
  });

  $("recs-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("recs-error", "");
    const userId = $("rec-user-id").value.trim();
    const method = $("rec-method").value;
    const topK = $("rec-top-k").value;
    const submit = $("recs-submit");
    submit.disabled = true;
    $("recs-status").textContent = "Loading recommendations…";
    try {
      const { payload, elapsed } = await requestJson(
        `/recommendations/user/${encodeURIComponent(userId)}?method=${encodeURIComponent(method)}&top_k=${topK}`
      );
      const methodLabel =
        payload.recommendation_mode === "user_hybrid"
          ? "Hybrid"
          : payload.recommendation_mode === "user_cf"
            ? "Collaborative"
            : payload.recommendation_mode === "user_content"
              ? "Content"
              : payload.recommendation_mode;
      renderMeta("recs-meta", [
        methodLabel,
        payload.fallback_reason ? `Fallback ${payload.fallback_reason}` : null,
        `Local request: ${elapsed} ms`,
      ]);
      renderCards("recs-results", payload.results, {
        emptyMessage: "No recommendations available for this user.",
      });
      $("recs-status").textContent = `${payload.results.length} result(s)`;
    } catch (error) {
      showError("recs-error", error.message);
      renderMeta("recs-meta", []);
      clearCards("recs-results");
      $("recs-status").textContent = "";
    } finally {
      submit.disabled = false;
    }
  });

  $("popular-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("popular-error", "");
    const topK = $("popular-top-k").value;
    const submit = $("popular-submit");
    submit.disabled = true;
    $("popular-status").textContent = "Loading popular products…";
    try {
      const { payload, elapsed } = await requestJson(`/recommendations/trending?top_k=${topK}`);
      renderMeta("popular-meta", ["Historical popularity", `Local request: ${elapsed} ms`]);
      renderCards("popular-results", payload.results, {
        variant: "popular",
        emptyMessage: "No popular products found.",
      });
      $("popular-status").textContent = `${payload.results.length} result(s)`;
    } catch (error) {
      showError("popular-error", error.message);
      renderMeta("popular-meta", []);
      clearCards("popular-results");
      $("popular-status").textContent = "";
    } finally {
      submit.disabled = false;
    }
  });

  retrievalMode.addEventListener("change", syncSearchControls);
  usePersonalization.addEventListener("change", syncSearchControls);
  document.querySelectorAll("[data-query]").forEach((button) => {
    button.addEventListener("click", () => {
      $("query").value = button.getAttribute("data-query");
      $("query").focus();
    });
  });

  syncSearchControls();
  loadReady();
})();
