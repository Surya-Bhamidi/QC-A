/**
 * app.js - Client-Side Logic for QViT Interactive Dashboard
 */

let currentSampleId = 0;
let customImageB64 = null;
let currentNoiseProb = 0.0;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadSampleSelector();
  initCircuitSliders();
  initNoiseSlider();
  initFaqAccordion();

  // Run initial prediction on sample 0
  setTimeout(() => {
    runPrediction();
  }, 400);
});

// Tab Switching
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      const activePane = document.getElementById(targetId);
      if (activePane) activePane.classList.add("active");
    });
  });
}

// Sample Loader
async function loadSampleSelector() {
  const container = document.getElementById("sample-picker-container");
  if (!container) return;

  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    const samples = data.samples || [];

    container.innerHTML = "";
    samples.forEach((sample, idx) => {
      const card = document.createElement("div");
      card.className = `sample-card ${idx === 0 ? "active" : ""}`;
      card.id = `sample-card-${sample.id}`;
      card.onclick = () => selectSample(sample.id);

      const tagClass = sample.label.toLowerCase();
      card.innerHTML = `
        <img class="sample-img-thumb" src="${sample.thumbnail_b64}" alt="Patient ${sample.id}" />
        <span class="sample-tag ${tagClass}">${sample.label}</span>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load test samples:", err);
  }
}

function selectSample(id) {
  currentSampleId = id;
  customImageB64 = null;

  document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
  const activeCard = document.getElementById(`sample-card-${id}`);
  if (activeCard) activeCard.classList.add("active");

  runPrediction();
}

// Custom Upload Handling
function handleImageUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    customImageB64 = e.target.result;
    currentSampleId = null;

    document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
    runPrediction();
  };
  reader.readAsDataURL(file);
}

// Run Live Prediction
async function runPrediction() {
  const btn = document.getElementById("btn-run-diagnosis");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="badge-live"></span> Analyzing Quantum State...`;
  }

  const payload = {
    noise_prob: currentNoiseProb,
  };
  if (customImageB64) {
    payload.image_b64 = customImageB64;
  } else {
    payload.sample_id = currentSampleId !== null ? currentSampleId : 0;
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await res.json();

    // Render results
    renderPredictionResults(result);
  } catch (err) {
    console.error("Inference failed:", err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `⚡ Run Quantum Diagnosis`;
    }
  }
}

function renderPredictionResults(data) {
  // Ground truth badge
  const gtEl = document.getElementById("ground-truth-badge");
  if (gtEl) {
    gtEl.textContent = data.ground_truth;
    gtEl.className = `verdict-badge ${data.ground_truth.toLowerCase()}`;
  }

  // Raw Image & Heatmap
  const rawImgEl = document.getElementById("display-raw-img");
  const overlayImgEl = document.getElementById("display-overlay-img");
  if (rawImgEl && data.raw_image) rawImgEl.src = data.raw_image;
  if (overlayImgEl && data.quantum.overlay_b64) overlayImgEl.src = data.quantum.overlay_b64;

  // QViT Results
  const qPred = data.quantum.prediction;
  const qConf = (data.quantum.confidence * 100).toFixed(1);
  const qBadge = document.getElementById("qvit-verdict-badge");
  const qBar = document.getElementById("qvit-conf-bar");
  const qConfText = document.getElementById("qvit-conf-text");
  const qLatency = document.getElementById("qvit-latency");

  if (qBadge) {
    qBadge.textContent = `${qPred} (${qConf}%)`;
    qBadge.className = `verdict-badge ${qPred.toLowerCase()}`;
  }
  if (qBar) {
    qBar.style.width = `${qConf}%`;
    qBar.className = `confidence-bar-fill ${qPred.toLowerCase()}`;
  }
  if (qConfText) qConfText.textContent = `Confidence: ${qConf}%`;
  if (qLatency) qLatency.textContent = `Latency: ${data.quantum.latency_ms}ms`;

  // Classical ViT Results
  const cPred = data.classical.prediction;
  const cConf = (data.classical.confidence * 100).toFixed(1);
  const cBadge = document.getElementById("classical-verdict-badge");
  const cBar = document.getElementById("classical-conf-bar");
  const cConfText = document.getElementById("classical-conf-text");
  const cLatency = document.getElementById("classical-latency");

  if (cBadge) {
    cBadge.textContent = `${cPred} (${cConf}%)`;
    cBadge.className = `verdict-badge ${cPred.toLowerCase()}`;
  }
  if (cBar) {
    cBar.style.width = `${cConf}%`;
    cBar.className = `confidence-bar-fill ${cPred.toLowerCase()}`;
  }
  if (cConfText) cConfText.textContent = `Confidence: ${cConf}%`;
  if (cLatency) cLatency.textContent = `Latency: ${data.classical.latency_ms}ms`;

  // Agreement Indicator
  const agreeBadge = document.getElementById("model-agreement-badge");
  if (agreeBadge) {
    if (qPred === cPred) {
      agreeBadge.textContent = "Concordant Diagnostic Agreement";
      agreeBadge.style.color = "var(--status-normal)";
    } else {
      agreeBadge.textContent = "Divergent Predictions";
      agreeBadge.style.color = "var(--status-warning)";
    }
  }
}

// Noise Slider Handling
function initNoiseSlider() {
  const slider = document.getElementById("noise-slider");
  const label = document.getElementById("noise-slider-val");
  if (!slider) return;

  slider.addEventListener("input", (e) => {
    currentNoiseProb = parseFloat(e.target.value);
    if (label) label.textContent = `${(currentNoiseProb * 100).toFixed(1)}%`;
  });

  slider.addEventListener("change", () => {
    runPrediction();
  });
}

// Interactive Quantum SWAP Test Calculator
function initCircuitSliders() {
  const qInputs = [
    document.getElementById("slider-q0"),
    document.getElementById("slider-q1"),
    document.getElementById("slider-q2"),
    document.getElementById("slider-q3"),
  ];
  const kInputs = [
    document.getElementById("slider-k0"),
    document.getElementById("slider-k1"),
    document.getElementById("slider-k2"),
    document.getElementById("slider-k3"),
  ];

  const updateCalc = async () => {
    const qVec = qInputs.map(inp => parseFloat(inp ? inp.value : 0.5));
    const kVec = kInputs.map(inp => parseFloat(inp ? inp.value : 0.4));

    // Update labels
    qInputs.forEach((inp, i) => {
      const el = document.getElementById(`val-q${i}`);
      if (el) el.textContent = qVec[i].toFixed(2);
    });
    kInputs.forEach((inp, i) => {
      const el = document.getElementById(`val-k${i}`);
      if (el) el.textContent = kVec[i].toFixed(2);
    });

    try {
      const res = await fetch("/api/swap_test_calc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: qVec, k: kVec }),
      });
      const data = await res.json();

      const fidEl = document.getElementById("swap-test-fidelity-val");
      if (fidEl) fidEl.textContent = data.state_fidelity.toFixed(5);

      const barEl = document.getElementById("swap-test-fidelity-bar");
      if (barEl) barEl.style.width = `${(data.state_fidelity * 100).toFixed(1)}%`;

      // Update per-qubit overlaps
      data.wire_fidelities.forEach((wf, idx) => {
        const wireEl = document.getElementById(`wire-fid-${idx}`);
        if (wireEl) wireEl.textContent = `F_${idx} = ${wf.toFixed(4)}`;
      });
    } catch (err) {
      console.error("SWAP test calculation error:", err);
    }
  };

  [...qInputs, ...kInputs].forEach(inp => {
    if (inp) inp.addEventListener("input", updateCalc);
  });

  // Initial calculation
  updateCalc();
}

// FAQ Accordion
function initFaqAccordion() {
  const items = document.querySelectorAll(".faq-item");
  items.forEach(item => {
    const question = item.querySelector(".faq-question");
    if (question) {
      question.addEventListener("click", () => {
        const isOpen = item.classList.contains("open");
        items.forEach(i => i.classList.remove("open"));
        if (!isOpen) item.classList.add("open");
      });
    }
  });
}
