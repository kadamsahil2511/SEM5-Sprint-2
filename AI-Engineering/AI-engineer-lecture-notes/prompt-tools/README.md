# Prompt tools

- [`compressor/`](compressor/) runs a local LLMLingua-2 prompt compression API.
- [`extension/`](extension/) is a Chrome extension that calls that API.
- [Image Studio](../../image-studio/) is a separate sibling project.

Install the shared Python environment once from the sprint root (see its
README). From this directory, activate it with `source ./activate.sh` on
macOS/Linux or `. .\activate.ps1` in PowerShell.

Run the compressor from its directory:

```sh
cd compressor
python app.py
```

It serves `http://127.0.0.1:7870`. Load the extension from the `extension/`
directory as described in its README.
