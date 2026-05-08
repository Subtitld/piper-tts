"""Subtitld add-on entry point for Piper TTS.

The build pipeline freezes this file (plus the `piper-tts` Python package) into
a single PyInstaller binary at `dist/piper-tts-<version>-<platform>.zip:bin/piper-addon`.
The host process (Subtitld) speaks JSON-line frames over our stdin/stdout —
see `subtitld.modules.addons.protocol` for the wire format.

Implementation notes:
  - Models live in `<addon_dir>/models/<voice_id>/{model.onnx,model.json}` by
    default. A user-overridable path can be supplied via the `models_dir`
    config field; we resolve `~` and env vars before opening.
  - We never block the stdout writer thread on Piper inference: the main
    thread reads requests, dispatches to a worker, and the worker emits
    `progress`/`result` frames as it makes progress.
  - Cancellation is best-effort. Piper's API is one synchronous call so we
    can't preempt mid-sentence; we just refuse new work after `cancel`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import wave
from pathlib import Path

log = logging.getLogger('piper-addon')
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[piper-addon] %(levelname)s %(message)s')

# Lazy import — keeps `--help` and dry-run cheap, and lets the host show a
# friendly error if Piper itself failed to import inside the frozen bundle.
try:
    from piper import PiperVoice  # type: ignore
except ImportError as exc:  # pragma: no cover
    print(json.dumps({
        'type': 'hello_error',
        'code': 'internal',
        'message': f'piper-tts python package not available: {exc}',
    }), flush=True)
    sys.exit(1)


PROTOCOL = 1
ADDON_ID = 'piper-tts'
VERSION = '1.0.1'


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()


def write_frame(frame: dict) -> None:
    line = json.dumps(frame, ensure_ascii=False)
    with _write_lock:
        sys.stdout.write(line + '\n')
        sys.stdout.flush()


def emit_progress(rid: str, value: float, message: str = '') -> None:
    write_frame({
        'id': rid, 'type': 'progress',
        'data': {'value': max(0.0, min(1.0, float(value))), 'message': message},
    })


def emit_error(rid: str, code: str, message: str, retryable: bool = False) -> None:
    write_frame({
        'id': rid, 'type': 'error',
        'data': {'code': code, 'message': message, 'retryable': retryable},
    })


def emit_result(rid: str, data: dict) -> None:
    write_frame({'id': rid, 'type': 'result', 'data': data})


# ---------------------------------------------------------------------------
# Voice handling
# ---------------------------------------------------------------------------
_voice_cache: dict[str, object] = {}
_voice_cache_lock = threading.Lock()


def _addon_root() -> Path:
    """Where the host installed this add-on (the directory holding
    `manifest.json` and `bin/`).

    The on-disk layout is:
        <addon_root>/
            manifest.json
            bin/
                piper-addon         ← exe (frozen) or just `python piper_addon.py` in dev
                _internal/...       ← PyInstaller `--onedir` payload (frozen only)
            models/<voice>/...

    We need a path that works in both contexts:
      - Frozen: `__file__` resolves into `bin/_internal/`, so going up two
        levels lands on the add-on root.
      - Source: `piper_addon.py` lives at the repo root during dev, so
        `Path(__file__).parent` IS the root.
    `sys.frozen` is the canonical PyInstaller marker.
    """
    if getattr(sys, 'frozen', False):
        # `sys.executable` is `<addon_root>/bin/piper-addon`; up two = root.
        return Path(sys.executable).resolve().parent.parent
    return Path(__file__).resolve().parent


def _models_dir() -> Path:
    env_override = os.environ.get('PIPER_ADDON_MODELS_DIR')
    if env_override:
        return Path(os.path.expandvars(os.path.expanduser(env_override)))
    return _addon_root() / 'models'


def _load_voice(voice_id: str):
    with _voice_cache_lock:
        cached = _voice_cache.get(voice_id)
        if cached is not None:
            return cached
        models_dir = _models_dir()
        candidates = [
            models_dir / voice_id / 'model.onnx',
            models_dir / f'{voice_id}.onnx',
        ]
        model_path = next((c for c in candidates if c.is_file()), None)
        if model_path is None:
            raise FileNotFoundError(
                f'voice {voice_id!r} not installed under {models_dir} '
                '(use the Subtitld voice installer)'
            )
        config_path = model_path.with_suffix('.onnx.json')
        if not config_path.is_file():
            config_path = model_path.with_suffix('.json')
        voice = PiperVoice.load(str(model_path), config_path=str(config_path) if config_path.is_file() else None)
        _voice_cache[voice_id] = voice
        return voice


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------
def handle_tts_synthesize(rid: str, params: dict) -> None:
    text = params.get('text')
    voice_id = params.get('voice')
    output_path = params.get('output_path')
    if not text or not voice_id or not output_path:
        emit_error(rid, 'bad_params', 'text, voice, and output_path are all required')
        return

    try:
        voice = _load_voice(voice_id)
    except FileNotFoundError as exc:
        emit_error(rid, 'model_missing', str(exc))
        return
    except Exception as exc:
        log.exception('voice load failed')
        emit_error(rid, 'internal', f'voice load failed: {exc}')
        return

    emit_progress(rid, 0.1, f'Loaded voice {voice_id}')

    # piper-tts ≥ 1.4 split the API: `synthesize()` is a generator yielding
    # AudioChunks, `synthesize_wav()` is the convenience wrapper that writes
    # them straight to a wave.Wave_write. We want the latter — same net
    # effect as the pre-1.4 `synthesize(text, wf)` call.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(output_path, 'wb') as wf:
            voice.synthesize_wav(text, wf)
    except Exception as exc:
        log.exception('synthesize failed')
        emit_error(rid, 'internal', f'synthesize failed: {exc}')
        return

    duration = 0.0
    sample_rate = 0
    channels = 0
    try:
        with wave.open(output_path, 'rb') as ro:
            sample_rate = ro.getframerate()
            channels = ro.getnchannels()
            frames = ro.getnframes()
            duration = frames / float(sample_rate or 1)
    except Exception:
        pass

    emit_result(rid, {
        'path': output_path,
        'duration_sec': duration,
        'sample_rate': sample_rate,
        'channels': channels,
    })


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    # Build voice/language inventory for the hello frame from the manifest
    # we ship inside the bundle. This avoids having to hardcode a parallel
    # list here.
    manifest_path = Path(__file__).resolve().parent / 'manifest.json'
    voices: list[dict] = []
    languages: list[str] = []
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            voices = manifest.get('voices') or []
            languages = sorted({v.get('language') for v in voices if v.get('language')})
        except Exception:
            log.exception('manifest parse failed')

    write_frame({
        'type': 'hello',
        'protocol': PROTOCOL,
        'addon': ADDON_ID,
        'version': VERSION,
        'capabilities': [
            {'task': 'tts.synthesize', 'languages': languages, 'voices': voices},
        ],
    })

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            frame = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            log.warning('skipping malformed frame: %s', exc)
            continue

        ftype = frame.get('type')
        rid = frame.get('id', '')

        if ftype == 'shutdown':
            log.info('shutdown received; exiting')
            return 0
        if ftype == 'cancel':
            # No-op — Piper synthesize is a single sync call.
            log.info('cancel ignored (sync inference)')
            continue
        if ftype == 'ready':
            # Host's handshake ack. Carries `host`/`host_version`; we don't
            # need either right now, but the frame must be tolerated rather
            # than treated as an unknown request — emitting an error here
            # would (a) write a frame without a valid `id` for the host to
            # route, and (b) suggest to a debugging human that something
            # is broken when it's actually the protocol working as designed.
            log.info('ready from host (host_version=%s)', frame.get('host_version'))
            continue
        if ftype == 'tts.synthesize':
            threading.Thread(
                target=handle_tts_synthesize,
                args=(rid, frame.get('params') or {}),
                daemon=True,
            ).start()
            continue

        # Unsolicited control frames (i.e. ones without an `id`) shouldn't
        # produce an error response — the host has nothing to correlate it
        # with. Log and move on; only request-shaped frames get a real
        # `bad_params` reply.
        if not rid:
            log.warning('ignoring unsolicited frame of type %r', ftype)
            continue

        emit_error(rid, 'bad_params', f'unknown request type: {ftype!r}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
