/**
 * Law Copilot - Document Ingestion & Pipeline Progress Handler
 * Real-time SSE streaming reader and interactive UI updates.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const selectedFileInfo = document.getElementById("selectedFileInfo");
  const uploadForm = document.getElementById("uploadForm");
  const submitBtn = document.getElementById("submitBtn");
  const resetBtn = document.getElementById("resetBtn");
  const docTitleInput = document.getElementById("docTitle");

  // Stepper & Progress Elements
  const stepperProgressLine = document.getElementById("stepperProgressLine");
  const progressBarFill = document.getElementById("progressBarFill");
  const progressPercentageText = document.getElementById("progressPercentageText");
  const currentStatusText = document.getElementById("currentStatusText");
  const stepNodes = [
    document.getElementById("stepNode1"),
    document.getElementById("stepNode2"),
    document.getElementById("stepNode3"),
    document.getElementById("stepNode4"),
    document.getElementById("stepNode5"),
  ];

  // Results & Extraction Elements
  const resultsSection = document.getElementById("resultsSection");
  const extractionBanner = document.getElementById("extractionBanner");
  const extractionBannerIcon = document.getElementById("extractionBannerIcon");
  const extractionBannerTitle = document.getElementById("extractionBannerTitle");
  const extractionBannerDesc = document.getElementById("extractionBannerDesc");
  const extractionStatusBadge = document.getElementById("extractionStatusBadge");
  const extractionStatusVal = document.getElementById("extractionStatusVal");
  const extractionPagesVal = document.getElementById("extractionPagesVal");
  const extractionCharsVal = document.getElementById("extractionCharsVal");
  const extractionTablesVal = document.getElementById("extractionTablesVal");
  const extractedTextSampleBox = document.getElementById("extractedTextSampleBox");
  const copySampleTextBtn = document.getElementById("copySampleTextBtn");

  // Chunking Elements
  const chunkingTotalBadge = document.getElementById("chunkingTotalBadge");
  const totalChunksStat = document.getElementById("totalChunksStat");
  const parentChunksStat = document.getElementById("parentChunksStat");
  const childChunksStat = document.getElementById("childChunksStat");
  const sampleChunksList = document.getElementById("sampleChunksList");

  // Console & Next Action Elements
  const eventLogConsole = document.getElementById("eventLogConsole");
  const clearLogBtn = document.getElementById("clearLogBtn");
  const nextActionCard = document.getElementById("nextActionCard");
  const uploadAnotherBtn = document.getElementById("uploadAnotherBtn");

  // --- Drag & Drop Setup ---
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(fileInput.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file) return;

    const sizeKb = (file.size / 1024).toFixed(1);
    selectedFileInfo.style.display = "inline-flex";
    selectedFileInfo.innerHTML = `📄 فایل انتخاب‌شده: <strong>${file.name}</strong> (${sizeKb} KB)`;

    // Auto-fill title if empty
    if (!docTitleInput.value.trim()) {
      const nameWithoutExt = file.name.substring(0, file.name.lastIndexOf(".")) || file.name;
      docTitleInput.value = nameWithoutExt.replace(/[-_]/g, " ");
    }

    logEvent("info", `فایل انتخاب شد: ${file.name} (${sizeKb} KB)`);
  }

  // --- Reset Form ---
  function resetAll() {
    uploadForm.reset();
    selectedFileInfo.style.display = "none";
    resultsSection.style.display = "none";
    nextActionCard.style.display = "none";
    updateProgress(0, "آماده برای بارگذاری سند جدید...");
    resetStepper();
    sampleChunksList.innerHTML = '<div style="text-align: center; padding: 2rem; color: var(--text-muted);">در حال انجام عملیات قطعه‌بندی...</div>';
    logEvent("info", "فرم بازنشانی شد.");
  }

  resetBtn.addEventListener("click", resetAll);
  if (uploadAnotherBtn) {
    uploadAnotherBtn.addEventListener("click", resetAll);
  }

  clearLogBtn.addEventListener("click", () => {
    eventLogConsole.innerHTML = "";
  });

  copySampleTextBtn.addEventListener("click", () => {
    const text = extractedTextSampleBox.innerText;
    if (text) {
      navigator.clipboard.writeText(text).then(() => {
        copySampleTextBtn.innerText = "کپی شد! ✓";
        setTimeout(() => (copySampleTextBtn.innerText = "کپی متن"), 2000);
      });
    }
  });

  // --- Stepper & Progress Utilities ---
  function resetStepper() {
    stepNodes.forEach((node) => {
      node.classList.remove("active", "completed");
    });
    stepperProgressLine.style.width = "0%";
  }

  function setStepState(stepIndex, state) {
    // stepIndex: 0..4
    for (let i = 0; i < stepIndex; i++) {
      stepNodes[i].classList.remove("active");
      stepNodes[i].classList.add("completed");
    }
    if (state === "active") {
      stepNodes[stepIndex].classList.add("active");
      stepNodes[stepIndex].classList.remove("completed");
    } else if (state === "completed") {
      stepNodes[stepIndex].classList.remove("active");
      stepNodes[stepIndex].classList.add("completed");
    }

    // Update stepper line width
    const percentage = Math.min(100, Math.max(0, (stepIndex / 4) * 100));
    stepperProgressLine.style.width = `${percentage}%`;
  }

  function updateProgress(percentage, message) {
    progressBarFill.style.width = `${percentage}%`;
    progressPercentageText.innerText = `${percentage.toLocaleString("fa-IR")}%`;
    if (message) {
      currentStatusText.innerText = message;
    }
  }

  function logEvent(type, message) {
    const entry = document.createElement("div");
    entry.className = `event-log-entry ${type}`;
    const timestamp = new Date().toLocaleTimeString("fa-IR");
    entry.innerText = `[${timestamp}] ${message}`;
    eventLogConsole.appendChild(entry);
    eventLogConsole.scrollTop = eventLogConsole.scrollHeight;
  }

  // --- Form Submission & SSE Streaming ---
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!fileInput.files || fileInput.files.length === 0) {
      alert("لطفاً ابتدا یک فایل را انتخاب یا در کادر رها کنید.");
      return;
    }

    const file = fileInput.files[0];
    const formData = new FormData(uploadForm);

    // Prepare UI
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>در حال پردازش...</span> <span class="status-dot"></span>`;
    resultsSection.style.display = "block";
    nextActionCard.style.display = "none";
    resetStepper();
    updateProgress(5, "در حال اتصال به سرور...");
    setStepState(0, "active");

    logEvent("info", `آغاز ارسال سند '${formData.get("title")}' به خط پردازش...`);

    try {
      const response = await fetch("/api/v1/pipeline/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`خطای سرور: کد وضعیت ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // keep last incomplete chunk

        for (const block of lines) {
          if (!block.trim()) continue;

          let eventType = "message";
          let dataStr = "";

          const blockLines = block.split("\n");
          for (const line of blockLines) {
            if (line.startsWith("event:")) {
              eventType = line.replace("event:", "").trim();
            } else if (line.startsWith("data:")) {
              dataStr += line.replace("data:", "").trim();
            }
          }

          if (dataStr) {
            try {
              const data = JSON.parse(dataStr);
              handlePipelineEvent(eventType, data);
            } catch (err) {
              console.error("JSON parse error:", err, dataStr);
            }
          }
        }
      }
    } catch (error) {
      logEvent("error", `خطا در ارتباط با سرور: ${error.message}`);
      updateProgress(100, `خطا: ${error.message}`);
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span>شروع بارگذاری و پردازش</span> <span>⚡</span>`;
    }
  });

  // --- Handle Pipeline Events ---
  function handlePipelineEvent(eventType, data) {
    const stage = data.stage;
    const progress = data.progress || 0;

    if (data.message) {
      updateProgress(progress, data.message);
    }

    if (eventType === "error") {
      logEvent("error", `خطا در مرحله ${stage}: ${data.error || data.message}`);
      extractionBanner.style.display = "flex";
      extractionBanner.className = "status-banner danger";
      extractionBannerIcon.innerText = "❌";
      extractionBannerTitle.innerText = "خطا در پردازش سند";
      extractionBannerDesc.innerText = data.error || data.message;
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span>شروع بارگذاری و پردازش</span> <span>⚡</span>`;
      return;
    }

    switch (stage) {
      case "upload":
        if (data.status === "started") {
          setStepState(0, "active");
          logEvent("info", `[گام ۱] در حال آپلود فایل '${data.filename}'...`);
        } else if (data.status === "completed") {
          setStepState(0, "completed");
          setStepState(1, "active");
          logEvent(
            "success",
            `[گام ۱] فایل با شناسه ${data.document_id} و حجم ${data.file_size_formatted} با موفقیت ثبت شد.`
          );
        }
        break;

      case "parsing":
        if (data.status === "started") {
          setStepState(1, "active");
          logEvent("info", "[گام ۲] استخراج متن از سند با موتور PyMuPDF آغاز شد...");
        } else if (data.status === "completed") {
          setStepState(1, "completed");
          setStepState(2, "active");

          const textExtracted = data.text_extracted;
          const pagesCount = data.pages_count;
          const totalChars = data.total_chars;

          // Update metrics display
          extractionPagesVal.innerText = pagesCount.toLocaleString("fa-IR");
          extractionCharsVal.innerText = totalChars.toLocaleString("fa-IR");
          extractionTablesVal.innerText = (data.tables_count || 0).toLocaleString("fa-IR");

          // Update Banner & Status based on whether text was extracted
          extractionBanner.style.display = "flex";
          if (textExtracted) {
            extractionBanner.className = "status-banner success";
            extractionBannerIcon.innerText = "✅";
            extractionBannerTitle.innerText = "متن سند با موفقیت استخراج شد";
            extractionBannerDesc.innerText = `تعداد ${pagesCount.toLocaleString(
              "fa-IR"
            )} صفحه پردازش شد و ${totalChars.toLocaleString(
              "fa-IR"
            )} کاراکتر متن خالص برای پایگاه دانش استخراج گردید.`;

            extractionStatusBadge.className = "badge badge-success";
            extractionStatusBadge.innerText = "استخراج موفقیت‌آمیز";
            extractionStatusVal.innerText = "استخراج کامل متن";
            extractionStatusVal.style.color = "var(--success-600)";
          } else {
            extractionBanner.className = "status-banner warning";
            extractionBannerIcon.innerText = "⚠️";
            extractionBannerTitle.innerText = "متنی از سند استخراج نشد (احتمالاً اسکن تصویری)";
            extractionBannerDesc.innerText =
              "فایل بارگذاری شده فاقد لایه متنی است یا شامل تصاویر اسکن شده بدون OCR می‌باشد.";

            extractionStatusBadge.className = "badge badge-gold";
            extractionStatusBadge.innerText = "فاقد متن متنی";
            extractionStatusVal.innerText = "بدون متن";
            extractionStatusVal.style.color = "var(--warning-500)";
          }

          // Sample text preview
          extractedTextSampleBox.innerText = data.sample_text || "(متنی برای نمایش یافت نشد)";

          logEvent(
            textExtracted ? "success" : "warn",
            `[گام ۲] استخراج متن: ${pagesCount} صفحه، ${totalChars} کاراکتر (وضعیت استخراج: ${
              textExtracted ? "موفق" : "ناموفق"
            })`
          );
        }
        break;

      case "normalization":
        if (data.status === "started") {
          setStepState(2, "active");
          logEvent("info", "[گام ۳] نرمال‌سازی ساختار سند و شناسایی مواد، تبصره‌ها و سرفصل‌ها...");
        } else if (data.status === "completed") {
          setStepState(2, "completed");
          setStepState(3, "active");

          logEvent(
            "success",
            `[گام ۳] نرمال‌سازی پایان یافت: ${data.blocks_count} بلوک ساختاری تفکیک گردید.`
          );
          if (data.headings && data.headings.length > 0) {
            logEvent("info", `سرفصل‌های شناسایی‌شده: ${data.headings.slice(0, 3).join(" | ")}...`);
          }
        }
        break;

      case "chunking":
        if (data.status === "started") {
          setStepState(3, "active");
          logEvent("info", "[گام ۴] قطعه‌بندی هوشمند آگاه از سرفصل‌ها با رعایت سقف ۳۰۰۰ کاراکتر...");
        } else if (data.status === "completed") {
          setStepState(3, "completed");
          setStepState(4, "active");

          totalChunksStat.innerText = data.total_chunks.toLocaleString("fa-IR");
          parentChunksStat.innerText = data.parent_chunks_count.toLocaleString("fa-IR");
          childChunksStat.innerText = data.child_chunks_count.toLocaleString("fa-IR");
          chunkingTotalBadge.innerText = `${data.total_chunks.toLocaleString("fa-IR")} قطعه تولید شد`;

          // Render Sample Chunks Cards
          renderSampleChunks(data.sample_chunks || []);

          logEvent(
            "success",
            `[گام ۴] قطعه‌بندی با موفقیت انجام شد: ${data.total_chunks} قطعه (${data.parent_chunks_count} والد، ${data.child_chunks_count} فرزند)`
          );
        }
        break;

      case "complete":
        setStepState(4, "completed");
        updateProgress(100, "پردازش سند و آماده‌سازی پایگاه دانش با موفقیت تکمیل شد!");
        logEvent("success", `[گام ۵] سند '${data.title}' با موفقیت در پایگاه دانش نمایه‌سازی شد.`);

        submitBtn.disabled = false;
        submitBtn.innerHTML = `<span>شروع بارگذاری و پردازش</span> <span>⚡</span>`;
        nextActionCard.style.display = "block";
        break;
    }
  }

  // --- Render Sample Chunks Helper ---
  function renderSampleChunks(chunks) {
    sampleChunksList.innerHTML = "";

    if (!chunks || chunks.length === 0) {
      sampleChunksList.innerHTML =
        '<div style="text-align: center; padding: 1.5rem; color: var(--text-muted);">قطعه‌ای برای نمایش یافت نشد.</div>';
      return;
    }

    chunks.forEach((chk, idx) => {
      const card = document.createElement("div");
      card.className = "chunk-card";

      const isParent = chk.chunk_type === "parent";
      const badgeClass = isParent ? "badge-gold" : "badge-primary";
      const typeLabel = isParent ? "قطعه والد (Parent)" : "قطعه فرزند (Child)";
      const sectionPathStr =
        chk.section_path && chk.section_path.length > 0
          ? chk.section_path.join(" › ")
          : chk.section || "بخش عمومی سند";

      const previewId = `chunkPreview_${idx}`;
      const fullId = `chunkFull_${idx}`;
      const toggleBtnId = `chunkToggleBtn_${idx}`;

      card.innerHTML = `
        <div class="chunk-header">
          <div class="chunk-meta">
            <span class="badge ${badgeClass}">${typeLabel}</span>
            <span class="badge badge-secondary">قطعه شماره ${(chk.chunk_index + 1).toLocaleString(
              "fa-IR"
            )}</span>
            <span class="badge badge-secondary">صفحه ${(chk.page || 1).toLocaleString("fa-IR")}</span>
            <span class="badge badge-secondary">${chk.char_count.toLocaleString(
              "fa-IR"
            )} کاراکتر</span>
          </div>
          <button type="button" class="btn btn-outline" id="${toggleBtnId}" style="padding: 0.2rem 0.6rem; font-size: 0.75rem;">
            نمایش کامل متن
          </button>
        </div>
        
        <div class="chunk-path" style="margin-bottom: 0.6rem;">
          <span>📍 مسیر سرفصل:</span>
          <span>${sectionPathStr}</span>
        </div>

        <div class="chunk-content" id="${previewId}">
          ${escapeHtml(chk.content_preview)}
        </div>
        <div class="chunk-content" id="${fullId}" style="display: none;">
          ${escapeHtml(chk.full_content)}
        </div>
      `;

      sampleChunksList.appendChild(card);

      // Wire toggle button
      const toggleBtn = document.getElementById(toggleBtnId);
      const previewEl = document.getElementById(previewId);
      const fullEl = document.getElementById(fullId);

      toggleBtn.addEventListener("click", () => {
        if (fullEl.style.display === "none") {
          fullEl.style.display = "block";
          previewEl.style.display = "none";
          toggleBtn.innerText = "بستن متن کامل";
        } else {
          fullEl.style.display = "none";
          previewEl.style.display = "block";
          toggleBtn.innerText = "نمایش کامل متن";
        }
      });
    });
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
