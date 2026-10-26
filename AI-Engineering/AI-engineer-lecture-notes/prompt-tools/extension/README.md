# Prompt Token Reducer (Chrome extension)

A Manifest V3 Chrome extension that shrinks long LLM prompts to fewer tokens.

Two modes:

- **Light mode (rules only)** &mdash; fast local pass: filler/politeness removal,
  verbose-phrase swaps (`in order to` &rarr; `to`), token-level abbreviations
  (`documentation` &rarr; `docs`, `maximum` &rarr; `max`, days/weeks &rarr; `d`/`w`),
  sentence/bullet dedup. 100% offline.
- **Deep mode (LLMLingua-2)** &mdash; calls the local Python service in
  [`../compressor/`](../compressor/) which runs Microsoft's **LLMLingua-2**
  model (xlm-roberta-large fine-tuned for prompt compression). Works on any
  prompt type &mdash; prose, image, code, instructions &mdash; because the
  model learned token importance from human labels. Falls back to the local
  rules if the server isn't running.

## Install

### Extension

1. Open `chrome://extensions`.
2. Enable **Developer mode** (top right).
3. Click **Load unpacked** and pick the `extension/` folder.
4. Pin the extension so the toolbar icon is visible.

### LLMLingua-2 server (for Deep mode)

```bash
cd ../compressor
source ../../../../.venv/bin/activate
python app.py
```

Serves on `http://127.0.0.1:7870`. See [compressor/README.md](../compressor/README.md).

## Use

- Click the toolbar icon to open the popup. Paste a prompt, hit **Reduce**,
  copy the result.
- On `chatgpt.com`, `claude.ai`, and `gemini.google.com`, a floating &#9889;
  button appears bottom-right. Focus a prompt box, click &#9889;, and the
  textarea is rewritten in place. Mode follows the popup toggle.

## Known limitations

- Token counts in the UI are approximate (`ceil(chars/4)`); real tokenizer
  counts will differ.
- Light-mode rules are English-tuned. LLMLingua-2 (Deep mode) is multilingual.
- First Deep-mode call downloads the LLMLingua-2 model (~2 GB) and takes a
  while; subsequent calls are fast.
