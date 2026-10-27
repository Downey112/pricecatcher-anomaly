# Full pipeline: build -> evaluate -> shortlist -> Power BI files -> tests. Run from the project root.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
& $py scripts\00_download.py; if ($LASTEXITCODE) { exit 1 }
& $py scripts\01_build.py;    if ($LASTEXITCODE) { exit 1 }
& $py scripts\02_evaluate.py; if ($LASTEXITCODE) { exit 1 }
& $py scripts\03_shortlist.py; if ($LASTEXITCODE) { exit 1 }
& $py scripts\05_export_bi.py; if ($LASTEXITCODE) { exit 1 }
& $py scripts\06_build_pbip.py; if ($LASTEXITCODE) { exit 1 }
& $py -m pytest tests -q
