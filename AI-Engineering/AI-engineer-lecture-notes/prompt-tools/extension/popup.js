// popup.js — wires up the popup UI.
// Deep mode uses the local LLMLingua-2 Python service when it's running;
// Light mode (and Deep-mode fallback) uses the built-in reducer.

const { countTokens, reduceRuleBased, reduceNLP } = self.PromptReducer;

const COMPRESSOR_URL = "http://127.0.0.1:7870";
const DEEP_RATE = 0.5; // fraction of tokens to keep in LLMLingua-2 compression.

const $ = (id) => document.getElementById(id);
const originalEl = $("original");
const reducedEl = $("reduced");
const countOrig = $("count-original");
const countRed = $("count-reduced");
const logEl = $("log");
const aggEl = $("aggressive");
const toggleLabelEl = $("toggleLabel");
const breakdownEl = $("breakdown");
const breakdownWrap = $("breakdownWrap");

function updateCounts() {
  countOrig.textContent = countTokens(originalEl.value);
  countRed.textContent = countTokens(reducedEl.value);
}

function updateToggleLabel() {
  toggleLabelEl.textContent = aggEl.checked
    ? "Deep mode (LLMLingua-2, local server)"
    : "Light mode (rules only)";
}

aggEl.addEventListener("change", async () => {
  updateToggleLabel();
  await chrome.storage.local.set({ aggressive: aggEl.checked });
});

originalEl.addEventListener("input", updateCounts);

function renderBreakdown(breakdown, savedTokens, originalTokens) {
  if (!breakdown) { breakdownWrap.style.display = "none"; return; }
  breakdownWrap.style.display = "";
  const rows = Object.entries(breakdown).filter(([, v]) => v !== 0);
  const labelW = 24;
  const lines = rows.map(([k, v]) => {
    const label = k.padEnd(labelW, " ");
    const sign = v > 0 ? "-" : "+";
    return `${label}${sign}${Math.abs(v)} tokens`;
  });
  lines.push("─".repeat(labelW + 12));
  const pct = originalTokens ? Math.round((savedTokens / originalTokens) * 100) : 0;
  lines.push(`${"Total".padEnd(labelW, " ")}-${savedTokens} tokens  (−${pct}%)`);
  breakdownEl.textContent = lines.join("\n");
}

async function compressRemote(text) {
  const ctrl = new AbortController();
  const to = setTimeout(() => ctrl.abort(), 60000);
  try {
    const res = await fetch(`${COMPRESSOR_URL}/api/compress`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text, rate: DEEP_RATE }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`server ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(to);
  }
}

$("reduce").addEventListener("click", async () => {
  const text = originalEl.value;
  if (!text.trim()) { logEl.textContent = "Nothing to reduce."; return; }

  const origTokens = countTokens(text);

  if (aggEl.checked) {
    logEl.textContent = "Compressing with LLMLingua-2 (local)…";
    try {
      const data = await compressRemote(text);
      reducedEl.value = data.compressed;
      updateCounts();
      const after = countTokens(data.compressed);
      const saved = origTokens - after;
      const pct = origTokens ? Math.round((saved / origTokens) * 100) : 0;
      logEl.textContent =
        `LLMLingua-2 · saved ~${saved} tokens (${pct}%). ` +
        `Model-reported ratio ${data.ratio || "n/a"}.`;
      renderBreakdown(
        { "LLMLingua-2 (model)": saved },
        saved,
        origTokens
      );
      return;
    } catch (err) {
      logEl.textContent =
        `LLMLingua-2 server unreachable (${err.message || err}).\n` +
        `Start it with: cd prompt-tools/compressor && python app.py\n` +
        `Falling back to local rules…`;
    }
  }

  try {
    const result = aggEl.checked ? reduceNLP(text) : reduceRuleBased(text);
    reducedEl.value = result.reduced;
    updateCounts();
    const pct = origTokens ? Math.round((result.savedTokens / origTokens) * 100) : 0;
    const header = result.savedTokens > 0
      ? `Saved ~${result.savedTokens} tokens (${pct}%).`
      : `No reductions found. Start the LLMLingua-2 server for real NLP compression.`;
    const existing = logEl.textContent ? logEl.textContent + "\n\n" : "";
    logEl.textContent = existing + header + (result.log.length ? "\n" + result.log.join("\n") : "");
    renderBreakdown(result.breakdown, result.savedTokens, origTokens);
  } catch (err) {
    logEl.textContent = "Error: " + (err.message || err);
    renderBreakdown(null);
  }
});

$("copy").addEventListener("click", async () => {
  if (!reducedEl.value) return;
  try {
    await navigator.clipboard.writeText(reducedEl.value);
    logEl.textContent = "Copied reduced prompt to clipboard.";
  } catch (err) {
    logEl.textContent = "Clipboard error: " + err.message;
  }
});

(async function init() {
  const { aggressive } = await chrome.storage.local.get(["aggressive"]);
  aggEl.checked = !!aggressive;
  updateToggleLabel();
  updateCounts();
  breakdownWrap.style.display = "none";
})();
