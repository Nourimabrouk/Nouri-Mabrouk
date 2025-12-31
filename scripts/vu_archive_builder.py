from __future__ import annotations

import os, json, tarfile, zipfile, hashlib, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import csv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from git import Repo
from git.exc import InvalidGitRepositoryError
from pygount import analysis as pyg_analysis
import pyarrow as pa
import pyarrow.parquet as pq

# Defaults
USER = "Nourimabrouk"
TODAY = datetime.now().strftime("%Y%m%d")
BASE = Path("D:/")
ROOT = BASE / f"Nouri_Mabrouk_VU_Code_Archive_{TODAY}"
API_BASE = os.environ.get("GITHUB_API_BASE", "https://api.github.com")

def shell(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept":"application/vnd.github+json","User-Agent":"vu-archive-builder/1.0"})
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok: s.headers["Authorization"] = f"Bearer {tok}"
    retry = Retry(total=5, read=5, connect=5, backoff_factor=1.5, status_forcelist=[429,500,502,503,504], allowed_methods=["GET","HEAD"], raise_on_status=False)
    ad = HTTPAdapter(max_retries=retry)
    s.mount("https://", ad); s.mount("http://", ad)
    return s

def paginated_get(s: requests.Session, url: str, params=None) -> List[Dict]:
    params = dict(params or {})
    out: List[Dict] = []
    while url:
        r = s.get(url, params=params, timeout=60)
        if r.status_code == 403 and "rate limit" in (r.text or '').lower():
            import time; time.sleep(10); continue
        if r.status_code >= 400: raise requests.HTTPError(f"GET {url} -> {r.status_code}")
        data = r.json(); out.extend(data if isinstance(data, list) else [data])
        link = r.headers.get('Link',''); nxt=None
        if link:
            for part in link.split(','):
                if 'rel="next"' in part:
                    nxt = part[part.find('<')+1:part.find('>')]; break
        url = nxt; params=None
    return out

def list_repos(s: requests.Session, user: str, include_archived=True) -> List[Dict]:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        try:
            url=f"{API_BASE}/user/repos"; params={"per_page":100,"type":"all","sort":"full_name","direction":"asc","affiliation":"owner,organization_member"}
            repos = paginated_get(s, url, params)
        except requests.HTTPError:
            url=f"{API_BASE}/users/{user}/repos"; params={"per_page":100,"type":"owner","sort":"full_name","direction":"asc"}
            repos = paginated_get(s, url, params)
    else:
        url=f"{API_BASE}/users/{user}/repos"; params={"per_page":100,"type":"owner","sort":"full_name","direction":"asc"}
        repos = paginated_get(s, url, params)
    if not include_archived: repos=[r for r in repos if not r.get('archived')]
    repos.sort(key=lambda r:(r.get('owner',{}).get('login',''), r.get('name','')))
    return repos

def safe(owner_repo: str) -> str: return owner_repo.replace('/', '__')

# --- Archive helpers ---
def repo_topics(sess: requests.Session, owner: str, name: str) -> List[str]:
    url=f"{API_BASE}/repos/{owner}/{name}/topics"
    r=sess.get(url, headers={"Accept":"application/vnd.github.mercy-preview+json"}, timeout=60)
    return r.json().get('names',[]) if r.status_code==200 else []

def repo_languages(sess: requests.Session, owner: str, name: str) -> Dict[str,int]:
    url=f"{API_BASE}/repos/{owner}/{name}/languages"; r=sess.get(url, timeout=60)
    return r.json() if r.status_code==200 else {}

def repo_license(sess: requests.Session, owner: str, name: str) -> Dict:
    url=f"{API_BASE}/repos/{owner}/{name}/license"; r=sess.get(url, timeout=60)
    return (r.json().get('license') or {}) if r.status_code==200 else {}

def make_mirror_and_bundle(clone_url: str, mirror_dir: Path) -> Tuple[Path, Path]:
    if mirror_dir.exists():
        try:
            repo=Repo(str(mirror_dir)); assert repo.bare
            repo.remotes.origin.fetch(prune=True)
        except Exception:
            shutil.rmtree(mirror_dir, ignore_errors=True)
            Repo.clone_from(clone_url, str(mirror_dir), mirror=True)
    else:
        Repo.clone_from(clone_url, str(mirror_dir), mirror=True)
    bundle = mirror_dir.with_suffix('.bundle')
    cp = shell(["git","bundle","create",str(bundle),"--all"], cwd=mirror_dir)
    if cp.returncode!=0: raise RuntimeError(f"git bundle failed: {cp.stderr[:300]}")
    return mirror_dir, bundle

def fetch_all_lfs_to_mirror(mirror_dir: Path) -> None:
    cp_ref=shell(["git","symbolic-ref","refs/remotes/origin/HEAD"], cwd=mirror_dir)
    default_ref=cp_ref.stdout.strip() if cp_ref.returncode==0 else "refs/remotes/origin/HEAD"
    branch=default_ref.rsplit('/',1)[-1] if '/' in default_ref else 'HEAD'
    temp_base = Path(mirror_dir.anchor) / "_vu_tmp"; temp_base.mkdir(parents=True, exist_ok=True)
    wt=temp_base/(mirror_dir.name+"_lfs_wt"); shutil.rmtree(wt, ignore_errors=True)
    add=shell(["git","worktree","add",str(wt),f"origin/{branch}"], cwd=mirror_dir)
    if add.returncode!=0:
        show=shell(["git","show-ref"], cwd=mirror_dir); ref=show.stdout.splitlines()[0].split()[1] if show.stdout else 'HEAD'
        shell(["git","worktree","add",str(wt),ref], cwd=mirror_dir)
    shell(["git","lfs","install"], cwd=wt)
    shell(["git","lfs","fetch","--all"], cwd=wt)
    shell(["git","worktree","remove","--force",str(wt)], cwd=mirror_dir)

def lfs_manifest_and_pack(mirror_dir: Path, out_dir: Path) -> Tuple[Path, Path]:
    out_manifest = out_dir/"lfs_manifest.csv"
    lfs_root = mirror_dir/"lfs"/"objects"
    temp_base = Path(mirror_dir.anchor) / "_vu_tmp"; temp_base.mkdir(parents=True, exist_ok=True)
    wt=temp_base/(mirror_dir.name+"_ls_wt"); shutil.rmtree(wt, ignore_errors=True)
    shell(["git","worktree","add",str(wt),"HEAD"], cwd=mirror_dir)
    shell(["git","lfs","install"], cwd=wt)
    ls=shell(["git","lfs","ls-files","-l"], cwd=wt)
    rows=[]
    if ls.returncode==0 and ls.stdout:
        for line in ls.stdout.splitlines():
            line=line.strip();
            if not line: continue
            oid=line.split()[0]; star=line.find("* "); rel=line[star+2:].strip() if star>=0 else ""
            size=""
            try:
                with open(wt/rel,'rb') as f:
                    head=f.read(512).decode('utf-8','ignore')
                for L in head.splitlines():
                    if L.startswith('size '): size=int(L.split()[1]); break
            except Exception: pass
            rows.append({"oid":oid,"size":size,"path":rel})
    shell(["git","worktree","remove","--force",str(wt)], cwd=mirror_dir)
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["oid","size","path"]) ; w.writeheader() ; w.writerows(rows)
    tarpath = out_dir/"lfs_objects.tar.gz"
    with tarfile.open(tarpath, 'w:gz') as tf:
        if lfs_root.exists():
            for p in lfs_root.rglob('*'):
                if p.is_file(): tf.add(p, arcname=str(p.relative_to(mirror_dir)))
    return out_manifest, tarpath

def commit_hist(mirror_dir: Path) -> Tuple[Optional[str],Optional[str],int,List[Dict[str,int]]]:
    repo=Repo(str(mirror_dir))
    try: _=repo.head.commit
    except Exception: return None,None,0,[]
    commits=list(repo.iter_commits("--all"))
    if not commits: return None,None,0,[]
    dates=[datetime.fromtimestamp(c.committed_date, tz=timezone.utc).date() for c in commits]
    first=min(dates).isoformat(); last=max(dates).isoformat()
    counts: Dict[int,int] = {}
    for d in dates:
        y=d.year; counts[y]=counts.get(y,0)+1
    rows=[{"year": int(y), "commits": int(counts[y])} for y in sorted(counts.keys())]
    return first,last,len(commits),rows

def sloc_from_worktree(mirror_dir: Path, out_dir: Path) -> Tuple[Path,Path,Dict[str,int],List[Dict[str,int]]]:
    temp_base = Path(mirror_dir.anchor) / "_vu_tmp"; temp_base.mkdir(parents=True, exist_ok=True)
    wt=temp_base/(mirror_dir.name+"_sloc_wt"); shutil.rmtree(wt, ignore_errors=True)
    add=shell(["git","worktree","add",str(wt),"HEAD"], cwd=mirror_dir)
    files_list: List[Dict[str,int]] = []
    lang_summary_list: List[Dict[str,int]] = []
    if add.returncode==0:
        # Walk the worktree and analyze source files
        skip_dirs = {'.git','__pycache__','.mypy_cache','.pytest_cache','.venv','venv','env','node_modules','dist','build','coverage','.idea','.vscode','.next'}
        results=[]
        for root, dirs, files in os.walk(wt):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fn in files:
                path = Path(root)/fn
                try:
                    a = pyg_analysis.SourceAnalysis.from_file(str(path), "pygount", encoding=None)
                    results.append(a)
                except Exception:
                    continue
        files_list = [
            {"path": os.path.relpath(r.filename, wt), "language": (r.language or 'Unknown'), "code": int(r.code_count), "comment": int(r.documentation_count), "empty": int(r.empty_count)}
            for r in results
        ] if results else []
        lang_acc: Dict[str, Dict[str,int]] = {}
        for row in files_list:
            lang=row["language"]
            acc=lang_acc.setdefault(lang, {"files":0, "code":0, "comment":0, "empty":0})
            acc["files"]+=1; acc["code"]+=int(row["code"]); acc["comment"]+=int(row["comment"]); acc["empty"]+=int(row["empty"])
        lang_summary_list = [{"language":k, **v} for k,v in sorted(lang_acc.items(), key=lambda kv: kv[1]['code'], reverse=True)]
    # write parquet via pyarrow
    fp=out_dir/"sloc_files.parquet"
    table = pa.Table.from_pylist(files_list) if files_list else pa.table({"path": [], "language": [], "code": [], "comment": [], "empty": []})
    pq.write_table(table, fp)
    # write per-language csv
    lc=out_dir/"sloc_by_language.csv"
    with open(lc, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["language","files","code","comment","empty"]) ; w.writeheader(); w.writerows(lang_summary_list)
    totals={"files":len(files_list), "code":sum(r.get('code',0) for r in files_list), "comment":sum(r.get('comment',0) for r in files_list), "empty":sum(r.get('empty',0) for r in files_list)}
    shell(["git","worktree","remove","--force",str(wt)], cwd=mirror_dir)
    return fp, lc, totals, lang_summary_list

def main(argv=None) -> int:
    import argparse
    p=argparse.ArgumentParser(description='Build VU-library–grade GitHub archive')
    p.add_argument('--user', default=USER); p.add_argument('--base', default=str(ROOT)); p.add_argument('--include-archived', action='store_true'); p.add_argument('--zip', action='store_true')
    a=p.parse_args(argv)
    root=Path(a.base); repos_dir=root/"data"/"repos"; meta_dir=root/"data"/"metadata"; logs_dir=root/"logs"
    for d in (repos_dir, meta_dir, logs_dir): d.mkdir(parents=True, exist_ok=True)
    print(f"\n==> Building VU-ready code archive at: {root}\n")
    sess=build_session()
    repos=list_repos(sess, a.user, include_archived=a.include_archived)
    if not repos:
        print(f"No repositories for {a.user}. If expecting private/org repos, set GITHUB_TOKEN."); return 1
    (meta_dir/"repos.inventory.json").write_text(json.dumps(repos, indent=2), encoding='utf-8')
    swh_links=[]; rows=[]; lang_totals:Dict[str,Dict[str,int]]={}
    for r in tqdm(repos, desc='Archiving repos'):
        owner=r.get('owner',{}).get('login'); name=r.get('name');
        if not owner or not name: continue
        owner_repo=f"{owner}/{name}"; safe_name=safe(owner_repo)
        try:
            repo_root=repos_dir/safe_name; bundle_dir=repo_root/"archive"; stats_dir=repo_root/"stats"
            repo_root.mkdir(parents=True, exist_ok=True); bundle_dir.mkdir(parents=True, exist_ok=True); stats_dir.mkdir(parents=True, exist_ok=True)
            mirror_dir=repo_root/f"{name}.git"
            try:
                mirror_dir,bundle=make_mirror_and_bundle(r.get('clone_url'), mirror_dir)
            except Exception as e:
                (logs_dir/f"{safe_name}_bundle_error.txt").write_text(str(e), encoding='utf-8'); continue
            try:
                fetch_all_lfs_to_mirror(mirror_dir)
            except Exception: pass
            lfs_manifest,lfs_tgz=lfs_manifest_and_pack(mirror_dir, bundle_dir)
            refs=bundle_dir/"refs.txt"; cp= shell(["git","show-ref"], cwd=mirror_dir); refs.write_text(cp.stdout or '', encoding='utf-8')
            first,last,count,hist_rows=commit_hist(mirror_dir)
            with open(stats_dir/"commits_by_year.csv", "w", newline="", encoding="utf-8") as f:
                w=csv.DictWriter(f, fieldnames=["year","commits"]) ; w.writeheader(); w.writerows(hist_rows)
            files_parq, lang_csv, totals, lang_summary_list = sloc_from_worktree(mirror_dir, stats_dir)
            for row in lang_summary_list:
                L=row['language']; acc=lang_totals.setdefault(L,{"files":0,"code":0,"comment":0,"empty":0})
                acc['files']+=int(row['files']); acc['code']+=int(row['code']); acc['comment']+=int(row['comment']); acc['empty']+=int(row['empty'])
            topics=repo_topics(sess, owner, name); gh_lang=repo_languages(sess, owner, name); lic=repo_license(sess, owner, name); spdx=lic.get('spdx_id') if isinstance(lic,dict) else ''
            meta={"repo":owner_repo,"description":r.get('description'),"visibility":"private" if r.get('private') else "public","is_fork":bool(r.get('fork')),
                  "is_archived":bool(r.get('archived')),"created_at":r.get('created_at'),"updated_at":r.get('updated_at'),"pushed_at":r.get('pushed_at'),
                  "default_branch":r.get('default_branch'),"stargazers_count":r.get('stargazers_count',0),"forks_count":r.get('forks_count',0),
                  "watchers_count":r.get('watchers_count',0),"size_kb_github":r.get('size',0),"first_commit_date":first,"last_commit_date":last,
                  "commit_count":count,"sloc_totals":totals,"topics":topics,
                  "top_languages_bytes_github":sorted(gh_lang.items(), key=lambda kv: kv[1], reverse=True)[:3],"license_spdx":spdx or ''}
            (repo_root/"repo_metadata.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')
            tar_path=repos_dir/f"{safe_name}.tar.gz"
            with tarfile.open(tar_path,'w:gz') as tf:
                tf.add(bundle, arcname=f"{safe_name}/archive/{bundle.name}"); tf.add(refs, arcname=f"{safe_name}/archive/{refs.name}")
                tf.add(repo_root/"repo_metadata.json", arcname=f"{safe_name}/repo_metadata.json")
                for p in stats_dir.rglob('*'): tf.add(p, arcname=f"{safe_name}/stats/{p.name}")
                if lfs_manifest.exists(): tf.add(lfs_manifest, arcname=f"{safe_name}/archive/{lfs_manifest.name}")
                if lfs_tgz.exists(): tf.add(lfs_tgz, arcname=f"{safe_name}/archive/{lfs_tgz.name}")
            (repo_root/"ARCHIVE_SHA256.txt").write_text(f"{sha256_file(tar_path)}  {tar_path.name}\n", encoding='utf-8')
            rows.append({k: meta.get(k) for k in ["repo","description","visibility","is_fork","is_archived","created_at","first_commit_date","last_commit_date","commit_count","default_branch","stargazers_count","forks_count","watchers_count","license_spdx"]} | {"total_files":totals['files'],"total_code":totals['code'],"total_comment":totals['comment'],"total_empty":totals['empty']})
            swh_links.append(f"https://archive.softwareheritage.org/save/https://github.com/{owner}/{name}/")
        except Exception as e:
            (logs_dir/f"{safe_name}_fatal_error.txt").write_text(str(e), encoding='utf-8')
            continue
    # repositories_summary CSV + Parquet
    rs_csv = meta_dir/"repositories_summary.csv"
    if rows:
        with open(rs_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames=list(rows[0].keys()); w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
        pq.write_table(pa.Table.from_pylist(rows), meta_dir/"repositories_summary.parquet")
    else:
        with open(rs_csv, "w", encoding="utf-8") as f: f.write("repo\n")
    # language_totals
    lrows=[{"language":k,**v} for k,v in sorted(lang_totals.items(), key=lambda kv: kv[1]['code'], reverse=True)]
    with open(meta_dir/"language_totals.csv", "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["language","files","code","comment","empty"]) ; w.writeheader(); w.writerows(lrows)
    if lrows:
        pq.write_table(pa.Table.from_pylist(lrows), meta_dir/"language_totals.parquet")
    readme=f"""# VU University Library Code Archive  Nouri Mabrouk ({TODAY})

This BagIt package contains archivist-grade snapshots of GitHub repositories under "{a.user}".

Included
--------
- data/repos/*.tar.gz — per-repository archival bundles with: *.bundle (full history), Git LFS objects (lfs_objects.tar.gz) + lfs_manifest.csv, repo_metadata.json, stats/ files
- data/metadata/repositories_summary.(csv|parquet) — per-repo overview
- data/metadata/language_totals.(csv|parquet) — totals across all repos
- data/metadata/repos.inventory.json — raw GitHub API inventory
- metadata.jsonld — schema.org/Code metadata (discovery)
- CITATION.cff — citation information
"""
    (root/"README_SNAPSHOT.md").write_text(readme, encoding='utf-8')
    citation=f"""cff-version: 1.2.0
message: "If you use this code archive, please cite it."
title: "Nouri Mabrouk  VU University Library Code Archive"
authors:
  - family-names: Mabrouk
    given-names: Nouri
identifiers:
  - type: url
    value: https://github.com/{a.user}
date-released: {datetime.now().strftime('%Y-%m-%d')}
"""
    (root/"CITATION.cff").write_text(citation, encoding='utf-8')
    jsonld={"@context":"https://schema.org","@type":"Collection","name":"VU University Library Code Archive  Nouri Mabrouk","creator":{"@type":"Person","name":"Nouri Mabrouk"},"dateCreated":datetime.now().strftime('%Y-%m-%d'),"isPartOf":"Vrije Universiteit Amsterdam Library","hasPart":[{"@type":"SoftwareSourceCode","codeRepository":f"https://github.com/{a.user}"}],"keywords":["Git archive","Software preservation","Git LFS","SLOC","BagIt"]}
    (root/"metadata.jsonld").write_text(json.dumps(jsonld, indent=2), encoding='utf-8')
    (meta_dir/"software_heritage_save_links.txt").write_text("\n".join(swh_links), encoding='utf-8')
    print("Creating BagIt manifests…")
    try:
        import bagit
        bagit.make_bag(str(root), {"Source-Organization":"Vrije Universiteit Amsterdam  University Library","Contact-Name":"Nouri Mabrouk","External-Description":"Archivist-grade GitHub code collection for long-term preservation.","Bagging-Date":datetime.now().strftime('%Y-%m-%d'),"Bag-Software-Agent":"vu_archive_builder.py"})
    except Exception:
        print("[WARN] bagit not installed; skipping BagIt wrapping (install via: py -m pip install bagit)")
    if a.zip:
        z = root.parent/f"{root.name}.zip"; 
        if z.exists(): z.unlink()
        print("Zipping BagIt directory…")
        with zipfile.ZipFile(z,'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in root.rglob('*'): zf.write(p, p.relative_to(root))
        (root/"BAG_ZIP_SHA256.txt").write_text(f"{sha256_file(z)}  {z.name}\n", encoding='utf-8')
        print(f"ZIP        : {z}")
    print("\nAll done."); print(f"Bag folder : {root}"); return 0

if __name__ == '__main__':
    import sys
    try:
        sys.exit(main())
    except requests.HTTPError as e:
        print(f"[HTTP ERROR] {e}"); sys.exit(2)
    except Exception as e:
        print(f"[FATAL] {e}"); sys.exit(1)
