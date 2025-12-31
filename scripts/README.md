VU Library Archive Builder

What
- Builds an archivist-grade snapshot of your GitHub account into a BagIt package, then zips it for handoff.
- Per-repo bundles include: full git history (.bundle), Git LFS objects (packed) + manifest, commit histograms, SLOC (file parquet + per-lang CSV), metadata JSON.
- Account-wide CSV/Parquet for repositories_summary and language_totals.

Install (once)
- py -m pip install --upgrade pip
- py -m pip install requests GitPython pygount pandas pyarrow tqdm python-dateutil bagit urllib3
- Optional: install Git and Git LFS; run: git lfs install
- Optional: setx GITHUB_TOKEN "ghp_your_token_here" (for private/org repos and higher limits)

Run
- py scripts\vu_archive_builder.py --user Nourimabrouk --zip
- Outputs to D:\Nouri_Mabrouk_VU_Code_Archive_YYYYMMDD and creates D:\Nouri_Mabrouk_VU_Code_Archive_YYYYMMDD.zip

Notes
- No shallow clones; full fidelity for preservation.
- If BagIt is not installed, the script will warn; install bagit and rerun to get manifests.
