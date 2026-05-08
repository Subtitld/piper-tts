# Piper TTS add-on for Subtitld

Lightweight neural TTS based on [Piper](https://github.com/rhasspy/piper).
Each voice is ~50 MB; runs entirely offline on CPU.

## Building

```bash
pip install piper-tts pyinstaller
pyinstaller piper-addon.spec --distpath dist/
cd dist/piper-addon
zip -r ../piper-tts-1.0.0-linux-x86_64.zip . ../../manifest.json ../../LICENSE ../../README.md
```

The resulting zip is what users (or the Subtitld catalog) install.

## Layout (post-install)

```
~/.local/share/subtitld/addons/piper-tts/
├── manifest.json
├── bin/piper-addon       # PyInstaller binary entry point
├── lib/...               # frozen Python + Piper deps
├── models/<voice>.onnx   # downloaded on first use of each voice
└── README.md
```

## Protocol

The host (Subtitld) speaks JSON-line frames over our stdin/stdout. See
`subtitld.modules.addons.protocol` for the wire spec. We support exactly one
task: `tts.synthesize`.

## License

The add-on glue (this repository's code) is MIT. Piper itself is MIT. Each
downloaded voice has its own license — most are CC-BY but some are CC0 or
custom — listed at <https://github.com/rhasspy/piper/blob/master/VOICES.md>.
