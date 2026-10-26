// content.js — floating ⚡ button that compresses the focused textarea.

(function () {
  if (window.__promptReducerInjected) return;
  window.__promptReducerInjected = true;

  const { reduceRuleBased, reduceNLP } = self.PromptReducer;

  let lastFocused = null;
  document.addEventListener(
    "focusin",
    (e) => {
      const t = e.target;
      if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text") || t.isContentEditable)) {
        lastFocused = t;
      }
    },
    true
  );

  const btn = document.createElement("button");
  btn.textContent = "⚡";
  btn.title = "Reduce prompt tokens";
  Object.assign(btn.style, {
    position: "fixed",
    right: "16px",
    bottom: "16px",
    width: "44px",
    height: "44px",
    borderRadius: "50%",
    border: "2px solid #1b1b1b",
    background: "#fffdf7",
    color: "#1b1b1b",
    cursor: "pointer",
    fontSize: "20px",
    zIndex: 2147483647,
    boxShadow: "2px 2px 0 #1b1b1b"
  });

  btn.addEventListener("click", async () => {
    const el = lastFocused || document.activeElement;
    if (!el) return;
    const text = el.isContentEditable ? el.innerText : el.value;
    if (!text || !text.trim()) return;

    try {
      const { aggressive } = await chrome.storage.local.get(["aggressive"]);
      let result;
      if (aggressive) {
        result = reduceNLP(text);
      } else {
        result = reduceRuleBased(text);
      }
      if (el.isContentEditable) {
        el.innerText = result.reduced;
      } else {
        el.value = result.reduced;
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }
      btn.textContent = "✓";
      setTimeout(() => (btn.textContent = "⚡"), 1000);
    } catch (err) {
      console.warn("[PromptReducer]", err);
      btn.textContent = "!";
      setTimeout(() => (btn.textContent = "⚡"), 1500);
    }
  });

  document.documentElement.appendChild(btn);
})();
