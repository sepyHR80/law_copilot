/**
 * Law Copilot - Persian Legal Chat Interface Handler
 * Coordinates query dispatching, markdown rendering, citations, and evidence display.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const queryInput = document.getElementById("queryInput");
  const sendBtn = document.getElementById("sendBtn");
  const typingIndicator = document.getElementById("typingIndicator");
  const newChatBtn = document.getElementById("newChatBtn");
  const filterKnowledgeType = document.getElementById("filterKnowledgeType");
  const filterDocType = document.getElementById("filterDocType");
  const topKInput = document.getElementById("topKInput");
  const enableExternalSearch = document.getElementById("enableExternalSearch");
  const currentIntentBadge = document.getElementById("currentIntentBadge");
  const sampleQueryButtons = document.querySelectorAll(".sample-query");

  // Intent translations in Persian
  const INTENT_TRANSLATIONS = {
    statute_lookup: "استعلام ماده قانونی",
    case_analysis: "تحلیل پرونده و دعوا",
    document_generation: "تنظیم و تدوین سند حقوقی",
    general_inquiry: "مشاوره عمومی حقوقی",
    unknown: "پرسش حقوقی",
  };

  // Auto-resize textarea
  queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = `${Math.min(queryInput.scrollHeight, 140)}px`;
  });

  // Enter to send (Shift+Enter for newline)
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Sample Query Buttons
  sampleQueryButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-query");
      if (q) {
        queryInput.value = q;
        queryInput.style.height = "auto";
        queryInput.style.height = `${Math.min(queryInput.scrollHeight, 140)}px`;
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

  // New Chat Button
  newChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = `
      <div class="message-bubble assistant">
        <div class="bubble-meta">
          <span>🤖 دستیار هوشمند حقوقی</span>
          <span>•</span>
          <span>هم‌اکنون</span>
        </div>
        <div class="bubble-body">
          <p style="margin-bottom: 0.5rem;"><strong>گفتگوی جدید آغاز شد.</strong></p>
          <p>پرسش حقوقی خود را بفرمایید تا مستند به مواد قانون و پایگاه دانش پاسخ دهم.</p>
        </div>
      </div>
    `;
    currentIntentBadge.style.display = "none";
    queryInput.value = "";
    queryInput.focus();
  });

  // Form Submit Handler
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const query = queryInput.value.trim();
    if (!query) return;

    // Append user message to chat
    appendUserMessage(query);
    queryInput.value = "";
    queryInput.style.height = "auto";

    // Show typing state
    setLoading(true);

    // Prepare Request Payload
    const filters = {};
    if (filterKnowledgeType.value) {
      filters.knowledge_type = filterKnowledgeType.value;
    }
    if (filterDocType.value) {
      filters.document_type = filterDocType.value;
    }

    const payload = {
      query: query,
      top_k: parseInt(topKInput.value) || 5,
      enable_external_search: enableExternalSearch.checked,
      filters: Object.keys(filters).length > 0 ? filters : null,
    };

    try {
      // 1. Try primary agent endpoint
      let response = await fetch("/api/v1/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      // 2. If agent endpoint fails or returns error, try fallback to RAG endpoint
      if (!response.ok) {
        const ragPayload = {
          question: query,
          top_k: parseInt(topKInput.value) || 5,
          filters: payload.filters,
        };

        const ragResponse = await fetch("/api/v1/rag/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(ragPayload),
        });

        if (ragResponse.ok) {
          const ragData = await ragResponse.json();
          renderAssistantResponse({
            query: ragData.question,
            response: ragData.answer,
            intent: "statute_lookup",
            citations: ragData.citations || [],
            evidence: ragData.evidence || [],
            is_sufficient: ragData.is_sufficient !== false,
          });
          return;
        } else {
          // If both backend services returned an error (e.g. no OpenAI API key in local environment)
          const errDetail = await response.json().catch(() => ({ detail: "خطای سرور" }));
          renderErrorMessage(
            `خطا در برقراری ارتباط با مدل هوش مصنوعی: ${errDetail.detail || response.statusText}.
            \n\n*نکته: در صورتی که کلید API مدل LLM در محیط تعریف نشده باشد، می‌توانید فایل‌های نمونه را ابتدا در بخش "بارگذاری و پردازش پایگاه دانش" بررسی نمایید.*`
          );
          return;
        }
      }

      const data = await response.json();
      renderAssistantResponse(data);
    } catch (err) {
      console.error("Chat error:", err);
      renderErrorMessage(`خطا در ارسال پیام: ${err.message}`);
    } finally {
      setLoading(false);
    }
  });

  // Append User Message to UI
  function appendUserMessage(text) {
    const bubble = document.createElement("div");
    bubble.className = "message-bubble user";
    const time = new Date().toLocaleTimeString("fa-IR", { hour: "2-digit", minute: "2-digit" });

    bubble.innerHTML = `
      <div class="bubble-meta">
        <span>👤 شما</span>
        <span>•</span>
        <span>${time}</span>
      </div>
      <div class="bubble-body">
        ${escapeHtml(text)}
      </div>
    `;

    chatMessages.appendChild(bubble);
    scrollToBottom();
  }

  // Render Assistant Message with Markdown, Citations & Evidence
  function renderAssistantResponse(data) {
    const bubble = document.createElement("div");
    bubble.className = "message-bubble assistant";
    const time = new Date().toLocaleTimeString("fa-IR", { hour: "2-digit", minute: "2-digit" });

    // Intent badge
    const intentFarsi = INTENT_TRANSLATIONS[data.intent] || data.intent || "پاسخ حقوقی";
    currentIntentBadge.style.display = "inline-flex";
    currentIntentBadge.innerText = `قصد: ${intentFarsi}`;

    // Sufficiency status
    const isSufficient = data.is_sufficient !== false;
    const sufficiencyBadge = isSufficient
      ? `<span class="badge badge-success">✓ مستند به شواهد قانونی</span>`
      : `<span class="badge badge-gold">⚠️ نیاز به استعلام تکمیلی</span>`;

    // Parse Markdown safely
    let formattedText = "";
    if (window.marked) {
      formattedText = window.marked.parse(data.response || "");
    } else {
      formattedText = `<p>${escapeHtml(data.response || "")}</p>`;
    }

    // Build Citations HTML
    let citationsHtml = "";
    if (data.citations && data.citations.length > 0) {
      const citationItems = data.citations
        .map((c, i) => {
          const sectionLabel = c.section ? `بخش: ${c.section}` : `مستند قانونی #${i + 1}`;
          const pageLabel = c.page ? ` (صفحه ${c.page.toLocaleString("fa-IR")})` : "";
          const snippetText = c.snippet ? escapeHtml(c.snippet) : "";

          return `
            <div style="background-color: var(--primary-50); border: 1px solid var(--primary-100); border-radius: var(--radius-sm); padding: 0.6rem 0.8rem; margin-top: 0.4rem; font-size: 0.82rem;">
              <div style="font-weight: 700; color: var(--primary-700); margin-bottom: 0.2rem;">
                📌 ${sectionLabel}${pageLabel}
              </div>
              ${snippetText ? `<div style="color: var(--text-main); font-size: 0.8rem; line-height: 1.6;">${snippetText}</div>` : ""}
            </div>
          `;
        })
        .join("");

      citationsHtml = `
        <div class="citations-box">
          <div class="citations-title">
            <span>📚</span>
            <span>استنادات و مراجع قانونی پاسخ (${data.citations.length.toLocaleString("fa-IR")} مورد):</span>
          </div>
          ${citationItems}
        </div>
      `;
    }

    // Build Evidence Items Toggle
    let evidenceHtml = "";
    if (data.evidence && data.evidence.length > 0) {
      const evidenceDetailsId = `evidenceDetails_${Date.now()}`;
      const evidenceItems = data.evidence
        .map((ev, i) => {
          return `
            <div style="padding: 0.6rem; border-bottom: 1px solid var(--surface-border); font-size: 0.8rem;">
              <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <strong>شاهد #${(i + 1).toLocaleString("fa-IR")} ${ev.section ? `(${ev.section})` : ""}</strong>
                <span class="badge badge-secondary">امتیاز: ${ev.score ? ev.score.toFixed(3) : "-"}</span>
              </div>
              <div style="color: var(--text-muted); line-height: 1.5;">${escapeHtml(ev.content ? ev.content.slice(0, 300) : "")}...</div>
            </div>
          `;
        })
        .join("");

      evidenceHtml = `
        <div style="margin-top: 0.75rem;">
          <details style="font-size: 0.82rem; color: var(--text-muted); cursor: pointer;">
            <summary style="font-weight: 600; color: var(--primary-600); user-select: none;">
              🔍 مشاهده شواهد و قطعات بازیابی‌شده از پایگاه دانش (${data.evidence.length.toLocaleString("fa-IR")} مورد)
            </summary>
            <div style="margin-top: 0.5rem; background: var(--surface-bg); border-radius: var(--radius-sm); border: 1px solid var(--surface-border); max-height: 250px; overflow-y: auto;">
              ${evidenceItems}
            </div>
          </details>
        </div>
      `;
    }

    bubble.innerHTML = `
      <div class="bubble-meta">
        <span>🤖 دستیار هوشمند حقوقی</span>
        <span>•</span>
        <span>${time}</span>
        <span>•</span>
        <span class="badge badge-primary">${intentFarsi}</span>
        ${sufficiencyBadge}
      </div>
      <div class="bubble-body">
        <div class="markdown-content" style="line-height: 1.75;">
          ${formattedText}
        </div>
        ${citationsHtml}
        ${evidenceHtml}
      </div>
    `;

    chatMessages.appendChild(bubble);
    scrollToBottom();
  }

  // Render Error Message in Chat
  function renderErrorMessage(errorText) {
    const bubble = document.createElement("div");
    bubble.className = "message-bubble assistant";
    const time = new Date().toLocaleTimeString("fa-IR", { hour: "2-digit", minute: "2-digit" });

    let formattedText = "";
    if (window.marked) {
      formattedText = window.marked.parse(errorText);
    } else {
      formattedText = `<p>${escapeHtml(errorText)}</p>`;
    }

    bubble.innerHTML = `
      <div class="bubble-meta">
        <span>🤖 دستیار هوشمند حقوقی</span>
        <span>•</span>
        <span>${time}</span>
        <span>•</span>
        <span class="badge badge-secondary" style="background-color: var(--danger-50); color: var(--danger-600);">خطا در پردازش</span>
      </div>
      <div class="bubble-body" style="border-color: rgba(239, 68, 68, 0.3); background-color: #fff9f9;">
        <div style="color: var(--danger-600);">
          ${formattedText}
        </div>
      </div>
    `;

    chatMessages.appendChild(bubble);
    scrollToBottom();
  }

  function setLoading(isLoading) {
    if (isLoading) {
      sendBtn.disabled = true;
      typingIndicator.style.display = "block";
    } else {
      sendBtn.disabled = false;
      typingIndicator.style.display = "none";
    }
    scrollToBottom();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
