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
import urllib.error
import urllib.request
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
VERSION = '1.0.2'

# rhasspy's piper-voices repo on HuggingFace lays files out as
#   <family>/<locale>/<name>/<quality>/<voice_id>.onnx
#   <family>/<locale>/<name>/<quality>/<voice_id>.onnx.json
# where voice_id is `<locale>-<name>-<quality>`, family is the language
# component of the locale (`en` from `en_US`, `pt` from `pt_BR`), etc.
# We derive URLs from this convention rather than reading them from the
# manifest because (a) the manifest only spells two of the six bundled
# voices out, and (b) it's stable enough that hardcoding it doesn't add
# much risk vs. the gain of zero-config voice fetching.
PIPER_VOICES_BASE = 'https://huggingface.co/rhasspy/piper-voices/resolve/main'


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


# Per-voice download lock. Two simultaneous synth requests for the same
# voice would otherwise race on the same .tmp file. The outer lock just
# guards `_download_locks` itself — actual downloads serialize on the
# inner per-voice lock so different voices can fetch in parallel.
_download_locks_guard = threading.Lock()
_download_locks: dict[str, threading.Lock] = {}


def _voice_id_parts(voice_id: str) -> tuple[str, str, str, str]:
    """Crack `<locale>-<name>-<quality>` apart, returning
    `(family, locale, name, quality)`.

    Voice names with internal `-`s are tolerated by joining the middle
    chunks back together (`en_GB-southern-english-medium` →
    name='southern-english'). Raises ValueError on a malformed id."""
    parts = voice_id.split('-')
    if len(parts) < 3 or '_' not in parts[0]:
        raise ValueError(
            f'voice id {voice_id!r} is not in <locale>-<name>-<quality> form'
        )
    locale = parts[0]
    quality = parts[-1]
    name = '-'.join(parts[1:-1])
    family = locale.split('_', 1)[0]
    return family, locale, name, quality


def _voice_url(voice_id: str, suffix: str) -> str:
    family, locale, name, quality = _voice_id_parts(voice_id)
    return f'{PIPER_VOICES_BASE}/{family}/{locale}/{name}/{quality}/{voice_id}.{suffix}'


def _download_with_progress(url: str, dest: Path, on_progress) -> None:
    """Stream `url` to `dest`, atomic-renaming from `<dest>.tmp` once
    complete. `on_progress(bytes_done, bytes_total)` fires after each
    chunk; pass a no-op when you don't care.

    Atomic rename matters because a partial download must not be picked
    up as installed by `_load_voice` on a later request — partial onnx
    files crash PiperVoice.load with cryptic onnxruntime errors."""
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        # `timeout=30` covers connect + per-read; HF's CDN has been
        # reliable so we don't bother with retries here.
        with urllib.request.urlopen(url, timeout=30) as resp:
            total = int(resp.headers.get('Content-Length') or 0)
            done = 0
            chunk = 256 * 1024  # 256 KiB; chosen to keep progress frames sub-1Hz on broadband
            with open(tmp, 'wb') as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    if total:
                        on_progress(done, total)
        os.replace(tmp, dest)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        # Clean up the .tmp file so the next attempt starts fresh.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _ensure_voice_files(voice_id: str, on_progress) -> tuple[Path, Path]:
    """Make sure both `<voice_id>.onnx` and `<voice_id>.onnx.json` exist
    under `_models_dir()`, downloading from HuggingFace if needed.

    `on_progress(value, message)` is called with `value` ∈ [0.0, 0.6]
    while downloading — leaves the 0.6→1.0 budget for the load+synth
    steps that follow in `handle_tts_synthesize`. Returns the resolved
    paths once both files are present."""
    with _download_locks_guard:
        lock = _download_locks.setdefault(voice_id, threading.Lock())

    with lock:
        models_dir = _models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f'{voice_id}.onnx'
        config_path = models_dir / f'{voice_id}.onnx.json'

        # Fast path: already present from a previous run.
        if model_path.is_file() and config_path.is_file():
            return model_path, config_path

        # Config first — it's tiny (kilobytes), so we don't bother
        # surfacing progress for it. If THIS download fails, the URL
        # pattern is wrong / voice doesn't exist on HF, and we save the
        # user from waiting on the 60 MB onnx.
        if not config_path.is_file():
            on_progress(0.0, f'Resolving {voice_id} on huggingface.co')
            try:
                _download_with_progress(
                    _voice_url(voice_id, 'onnx.json'),
                    config_path,
                    lambda _d, _t: None,
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise FileNotFoundError(
                        f'voice {voice_id!r} not found on rhasspy/piper-voices'
                    ) from exc
                raise

        # Now the big onnx. Map raw bytes-progress (0..1) into our
        # `0.0..0.6` slice of the request's overall progress.
        if not model_path.is_file():
            on_progress(0.0, f'Downloading {voice_id} model')
            def _scaled(done: int, total: int) -> None:
                frac = done / total if total else 0.0
                mb_done = done // (1024 * 1024)
                mb_total = total // (1024 * 1024)
                on_progress(
                    0.6 * frac,
                    f'Downloading {voice_id}: {mb_done}/{mb_total} MB',
                )
            _download_with_progress(
                _voice_url(voice_id, 'onnx'),
                model_path,
                _scaled,
            )

        return model_path, config_path


def _load_voice(voice_id: str, on_progress=None):
    """Cached PiperVoice loader. Triggers a download from HF the first
    time a voice is requested (see `_ensure_voice_files`).

    `on_progress(value, message)` is plumbed through to the downloader
    so the caller can forward it as protocol `progress` frames. Pass
    `None` if you don't care (e.g. a CLI test path)."""
    # Cache check OUTSIDE the load lock so already-loaded voices skip
    # all the file checks.
    with _voice_cache_lock:
        cached = _voice_cache.get(voice_id)
        if cached is not None:
            return cached

    cb = on_progress or (lambda *_: None)
    model_path, config_path = _ensure_voice_files(voice_id, cb)

    # Re-check the cache before constructing — two concurrent first-load
    # requests may have raced past the initial check; let the loser reuse
    # the winner's PiperVoice instead of paying for a second `.load()`.
    with _voice_cache_lock:
        cached = _voice_cache.get(voice_id)
        if cached is not None:
            return cached
        voice = PiperVoice.load(str(model_path), config_path=str(config_path))
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
        # Forward downloader progress straight to the host. The downloader
        # already constrains its values to the 0.0..0.6 range — the rest
        # of this function bumps to 0.7 (loaded) and 1.0 (synthesized).
        voice = _load_voice(
            voice_id,
            on_progress=lambda value, message: emit_progress(rid, value, message),
        )
    except (FileNotFoundError, ValueError) as exc:
        emit_error(rid, 'model_missing', str(exc))
        return
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        emit_error(rid, 'model_missing',
                   f'failed to download voice {voice_id!r}: {exc}')
        return
    except Exception as exc:
        log.exception('voice load failed')
        emit_error(rid, 'internal', f'voice load failed: {exc}')
        return

    emit_progress(rid, 0.7, f'Loaded voice {voice_id}')

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
