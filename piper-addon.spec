# PyInstaller spec for piper-tts add-on.
# Build with: pyinstaller piper-addon.spec --distpath dist/
# The resulting `dist/piper-addon/` directory + manifest.json + LICENSE +
# README.md are zipped into `piper-tts-<version>-<platform>.zip` for the
# Subtitld add-on catalog.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules('piper')
datas = collect_data_files('piper')

block_cipher = None

a = Analysis(
    ['piper_addon.py'],
    pathex=[],
    binaries=[],
    datas=datas + [('manifest.json', '.')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
