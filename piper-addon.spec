# PyInstaller spec for the piper-tts add-on.
# Build with: pyinstaller piper-addon.spec --distpath dist/
# The resulting `dist/piper-addon/` directory + manifest.json + LICENSE +
# README.md are packed by the release workflow into
# `piper-tts-<version>-<platform>.zip` for the Subtitld add-on catalog.

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# `piper-tts ≥ 1.4` ships:
#   - python sources under piper/
#   - a native helper:           piper/espeakbridge.so   (Linux/macOS) /
#                                piper/espeakbridge.pyd  (Windows)
#   - hundreds of dictionaries:  piper/espeak-ng-data/<lang>_dict
# `collect_data_files` grabs all non-`.py` files in the package, which
# *should* include both the dictionaries and the native lib — but on some
# platforms PyInstaller's classification puts shared libraries through the
# `binaries` channel instead (so they get stripped/rewritten properly).
# `collect_dynamic_libs` is the canonical way to ensure the .so is treated
# as a binary; using both is safe (PyInstaller dedups).
hiddenimports = (
    collect_submodules('piper')
    # onnxruntime has C extension submodules that aren't reachable via static
    # analysis of the addon code — without these the frozen import fails
    # the moment we ask for `onnxruntime.InferenceSession`.
    + collect_submodules('onnxruntime')
)
datas = collect_data_files('piper') + [('manifest.json', '.')]
binaries = collect_dynamic_libs('piper') + collect_dynamic_libs('onnxruntime')

block_cipher = None

a = Analysis(
    ['piper_addon.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # We don't use any of the heavy training-only deps; excluding them
    # keeps the bundle from accidentally pulling in torch via transitive
    # imports if the user's site-packages happens to have it.
    excludes=['torch', 'lightning', 'tensorboard', 'tensorboardX'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='piper-addon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, upx_exclude=[],
    name='piper-addon',
)
