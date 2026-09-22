/**
 * Frontend logic for AI Execution Traces and Chat Categorization Dashboard.
 */

document.addEventListener("DOMContentLoaded", () => {
  let activeCategory = "";
  let activeStatus = "";
  let activeSearch = "";
  let currentTraceData = null;

  // DOM Elements
  const statTotalTraces = document.getElementById("statTotalTraces");
  const statSuccessRate = document.getElementById("statSuccessRate");
  const statAvgLatency = document.getElementById("statAvgLatency");
  const statCategoriesCount = document.getElementById("statCategoriesCount");

  const categoriesList = document.getElementById("categoriesList");
  const allCountBadge = document.getElementById("allCountBadge");
  const searchInput = document.getElementById("searchInput");
  const statusFilter = document.getElementById("statusFilter");
  const refreshBtn = document.getElementById("refreshBtn");

  const currentFeedTitle = document.getElementById("currentFeedTitle");
  const feedCountLabel = document.getElementById("feedCountLabel");
  const tracesFeedContainer = document.getElementById("tracesFeedContainer");

  const traceModal = document.getElementById("traceModal");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const modalTraceId = document.getElementById("modalTraceId");
  const modalUserQuery = document.getElementById("modalUserQuery");
  const modalAiResponse = document.getElementById("modalAiResponse");
  const modalModelName = document.getElementById("modalModelName");
  const modalLatency = document.getElementById("modalLatency");
  const modalCategory = document.getElementById("modalCategory");
  const modalStatusPill = document.getElementById("modalStatusPill");
  const modalTimeline = document.getElementById("modalTimeline");
  const modalEvidenceContainer = document.getElementById("modalEvidenceContainer");
  const modalRawJson = document.getElementById("modalRawJson");
  const copyJsonBtn = document.getElementById("copyJsonBtn");

  // Initial load
  loadDashboard();

  // Check URL params for auto-opening a specific trace (e.g. ?id=...)
  const urlParams = new URLSearchParams(window.location.search);
  const targetTraceId = urlParams.get("id");
  if (targetTraceId) {
    openTraceDetail(targetTraceId);
  }

  // Refresh
  refreshBtn.addEventListener("click", () => {
    loadDashboard();
  });

  // Search input debounce
  let searchTimeout = null;
  searchInput.addEventListener("input", (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      activeSearch = e.target.value.trim();
      loadTraces();
    }, 350);
  });

  // Status filter
  statusFilter.addEventListener("change", (e) => {
    activeStatus = e.target.value;
    loadTraces();
  });

  // Close modal
  closeModalBtn.addEventListener("click", () => {
    traceModal.classList.remove("open");
  });
  window.addEventListener("click", (e) => {
    if (e.target === traceModal) {
      traceModal.classList.remove("open");
    }
  });

  // Copy JSON
  copyJsonBtn.addEventListener("click", () => {
    if (currentTraceData) {
      navigator.clipboard.writeText(JSON.stringify(currentTraceData, null, 2)).then(() => {
        copyJsonBtn.textContent = "✓ کپی شد!";
        setTimeout(() => {
          copyJsonBtn.textContent = "📋 کپی JSON";
        }, 2000);
      });
    }
  });

  async function loadDashboard() {
    await Promise.all([loadCategories(), loadTraces()]);
  }

  async function loadCategories() {
    try {
      const res = await fetch("/api/v1/traces/categories");
      if (!res.ok) throw new Error("Failed to load categories");
      const data = await res.json();

      statTotalTraces.textContent = Number(data.total_traces).toLocaleString("fa-IR");
      statSuccessRate.textContent = `${data.overall_success_rate}%`;
      statAvgLatency.textContent = `${Math.round(data.avg_system_latency_ms)} ms`;
      statCategoriesCount.textContent = Number(data.categories.length).toLocaleString("fa-IR");
      allCountBadge.textContent = data.total_traces;

      renderCategories(data.categories, data.total_traces);
    } catch (err) {
      console.error("Error loading categories:", err);
    }
  }

  function renderCategories(categories, totalCount) {
    // Preserve "All" item
    categoriesList.innerHTML = `
      <div class="category-item ${activeCategory === '' ? 'active' : ''}" data-category="">
        <span>همه گفتگوها</span>
        <span class="category-badge">${totalCount}</span>
      </div>
    `;

    categories.forEach((cat) => {
      const item = document.createElement("div");
      item.className = `category-item ${activeCategory === cat.category ? 'active' : ''}`;
      item.dataset.category = cat.category;
      item.innerHTML = `
        <span>${escapeHtml(cat.category)}</span>
        <span class="category-badge">${cat.count}</span>
      `;
      categoriesList.appendChild(item);
    });

    // Attach click events
    categoriesList.querySelectorAll(".category-item").forEach((el) => {
      el.addEventListener("click", () => {
        categoriesList.querySelectorAll(".category-item").forEach((c) => c.classList.remove("active"));
        el.classList.add("active");
        activeCategory = el.dataset.category;
        currentFeedTitle.textContent = activeCategory ? activeCategory : "همه گفتگوها";
        loadTraces();
      });
    });
  }

  async function loadTraces() {
    tracesFeedContainer.innerHTML = `
      <div style="text-align: center; padding: 3rem; color: var(--text-muted);">
        در حال بارگذاری لاگ‌ها...
      </div>
    `;

    const params = new URLSearchParams();
    if (activeCategory) params.append("category", activeCategory);
    if (activeStatus) params.append("status", activeStatus);
    if (activeSearch) params.append("search", activeSearch);
    params.append("limit", "50");

    try {
      const res = await fetch(`/api/v1/traces?${params.toString()}`);
      if (!res.ok) throw new Error("Failed to load traces");
      const data = await res.json();

      feedCountLabel.textContent = `نمایش ${data.items.length} از ${data.total} پیام`;

      if (data.items.length === 0) {
        tracesFeedContainer.innerHTML = `
          <div class="card" style="text-align: center; padding: 3rem; color: var(--text-muted);">
            پیامی با این مشخصات یافت نشد. می‌توانید با ارسال پرسش در بخش «دستیار گفتگو حقوقی» اولین لاگ‌ها را ثبت نمایید.
          </div>
        `;
        return;
      }

      tracesFeedContainer.innerHTML = "";
      data.items.forEach((item) => {
        const card = createTraceCard(item);
        tracesFeedContainer.appendChild(card);
      });
    } catch (err) {
      console.error("Error loading traces:", err);
      tracesFeedContainer.innerHTML = `
        <div class="card" style="text-align: center; padding: 2rem; color: var(--danger-600);">
          خطا در دریافت لیست لاگ‌ها: ${err.message}
        </div>
      `;
    }
  }

  function createTraceCard(item) {
    const card = document.createElement("div");
    card.className = "trace-card";

    let statusPillClass = "success";
    let statusLabel = "موفق و مستند";
    if (item.status === "insufficient" || !item.is_sufficient) {
      statusPillClass = "insufficient";
      statusLabel = "شواهد ناکافی";
    } else if (item.status === "error" || item.status === "rate_limited") {
      statusPillClass = "error";
      statusLabel = "خطا";
    }

    const dateStr = new Date(item.created_at).toLocaleString("fa-IR", {
      dateStyle: "short",
      timeStyle: "short",
    });

    card.innerHTML = `
      <div class="trace-header">
        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
          <span class="category-badge">${escapeHtml(item.category)}</span>
          <span class="status-pill ${statusPillClass}">${statusLabel}</span>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">${dateStr}</div>
      </div>

      <div class="trace-query">
        <span style="color: var(--primary-600);">پرسش:</span>
        <span>${escapeHtml(item.user_query)}</span>
      </div>

      <div class="trace-response-preview">
        ${escapeHtml(item.ai_response_preview || "بدون پاسخ")}
      </div>

      <div class="trace-meta-footer">
        <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
          <div>مدل: <strong>${escapeHtml(item.model_name || "gemini-3.5-flash-lite")}</strong></div>
          <div>زمان پاسخ: <strong>${item.latency_ms ? item.latency_ms + " ms" : "-"}</strong></div>
          <div>گام‌های طی‌شده: <strong>${item.step_count} مرحله</strong></div>
        </div>

        <button class="btn btn-secondary btn-sm view-trace-btn" style="padding: 0.35rem 0.85rem; font-size: 0.82rem;" data-id="${item.id}">
          🔍 مشاهده مسیر پردازش و دیباگ
        </button>
      </div>
    `;

    card.querySelector(".view-trace-btn").addEventListener("click", () => {
      openTraceDetail(item.id);
    });

    return card;
  }

  async function openTraceDetail(traceId) {
    try {
      const res = await fetch(`/api/v1/traces/${traceId}`);
      if (!res.ok) throw new Error("Trace not found");
      const trace = await res.json();
      currentTraceData = trace;

      modalTraceId.textContent = `ID: ${trace.id} | ${trace.created_at}`;
      modalUserQuery.textContent = trace.user_query;
      modalAiResponse.textContent = trace.ai_response;
      modalModelName.textContent = trace.model_name || "gemini-3.5-flash-lite";
      modalLatency.textContent = trace.latency_ms ? `${trace.latency_ms} میلی‌ثانیه` : "-";
      modalCategory.textContent = trace.category;

      let statusPillClass = "success";
      let statusLabel = "موفق و مستند به مواد قانونی";
      if (!trace.is_sufficient || trace.status === "insufficient") {
        statusPillClass = "insufficient";
        statusLabel = "شواهد ناکافی پایگاه دانش";
      } else if (trace.status === "error" || trace.status === "rate_limited") {
        statusPillClass = "error";
        statusLabel = "خطای مدل / سقف مصرف";
      }
      modalStatusPill.className = `status-pill ${statusPillClass}`;
      modalStatusPill.textContent = statusLabel;

      // Render Timeline
      renderTimeline(trace.execution_path || []);

      // Render Evidence
      renderEvidence(trace.evidence || trace.retrieval_data || []);

      // Render Raw JSON
      modalRawJson.textContent = JSON.stringify(trace, null, 2);

      traceModal.classList.add("open");
    } catch (err) {
      alert("خطا در باز کردن لاگ: " + err.message);
    }
  }

  function renderTimeline(steps) {
    if (!steps || steps.length === 0) {
      modalTimeline.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">اطلاعات گام‌به‌گام ثبت نشده است.</div>`;
      return;
    }

    modalTimeline.innerHTML = "";
    steps.forEach((s, idx) => {
      const item = document.createElement("div");
      item.className = "timeline-item";

      let dotClass = "";
      let dotIcon = "✓";
      if (s.status === "failed") {
        dotClass = "failed";
        dotIcon = "✕";
      } else if (s.status === "skipped") {
        dotClass = "skipped";
        dotIcon = "—";
      }

      const durText = s.duration_ms ? `${s.duration_ms} ms` : "۱ ms";

      let detailsHtml = "";
      if (s.details && Object.keys(s.details).length > 0) {
        detailsHtml = `
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.35rem; direction: ltr; text-align: left; background: #fff; padding: 0.5rem; border-radius: 6px; border: 1px solid var(--surface-border);">
            <pre style="margin: 0; white-space: pre-wrap; word-break: break-all; font-family: monospace;">${escapeHtml(JSON.stringify(s.details, null, 2))}</pre>
          </div>
        `;
      }

      item.innerHTML = `
        <div class="timeline-dot ${dotClass}">${dotIcon}</div>
        <div class="timeline-card">
          <div class="timeline-title">
            <span>گام ${idx + 1}: ${escapeHtml(s.title || s.step)}</span>
            <span class="timeline-duration">${durText}</span>
          </div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">
            کد مرحله: <code>${escapeHtml(s.step)}</code> | وضعیت: <strong>${s.status}</strong>
          </div>
          ${detailsHtml}
        </div>
      `;

      modalTimeline.appendChild(item);
    });
  }

  function renderEvidence(evidence) {
    if (!evidence || evidence.length === 0) {
      modalEvidenceContainer.innerHTML = `
        <div style="background: var(--surface-bg); padding: 1rem; border-radius: var(--radius-md); color: var(--text-muted); font-size: 0.85rem;">
          هیچ مدرکی در این تعامل بازیابی نشده یا نیازی به بازیابی نبوده است.
        </div>
      `;
      return;
    }

    modalEvidenceContainer.innerHTML = "";
    evidence.forEach((ev, idx) => {
      const box = document.createElement("div");
      box.style.cssText = "background: var(--surface-bg); border: 1px solid var(--surface-border); border-radius: var(--radius-md); padding: 0.85rem; margin-bottom: 0.75rem;";

      const title = ev.title || (ev.source && ev.source.title) || `سند شماره ${idx + 1}`;
      const score = ev.score !== undefined ? `امتیاز شباهت: ${ev.score}` : "";
      const page = ev.page ? `صفحه: ${ev.page}` : "";
      const content = ev.content || ev.content_preview || "";

      box.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 0.85rem; color: var(--primary-700); margin-bottom: 0.35rem;">
          <span>📌 ${escapeHtml(title)}</span>
          <span style="font-size: 0.78rem; color: var(--text-muted);">${page} ${score}</span>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-main); line-height: 1.5; background: #fff; padding: 0.65rem; border-radius: 6px; border: 1px solid var(--surface-border);">
          ${escapeHtml(content)}
        </div>
      `;
      modalEvidenceContainer.appendChild(box);
    });
  }

  function escapeHtml(text) {
    if (!text) return "";
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
