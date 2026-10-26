// background.js — service worker. Minimal; most work happens in popup/content.

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(["aggressive"]).then(({ aggressive }) => {
    if (typeof aggressive === "undefined") {
      chrome.storage.local.set({ aggressive: false });
    }
  });
});
