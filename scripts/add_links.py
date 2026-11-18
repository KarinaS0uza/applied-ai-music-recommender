"""Add exact youtube_url and spotify_url columns to data/songs.csv.

Resumable: every lookup is cached in a JSON file, so a rerun only does missing work.

    python3 scripts/add_links.py youtube          # no key needed (reads public result pages)
    python3 scripts/add_links.py spotify          # needs SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
    python3 scripts/add_links.py write            # merge the cache into data/songs.csv + report

Matching is deliberately strict: a link is only accepted when the title AND an artist name
match, so a song that cannot be found stays empty instead of getting a wrong link.
"""
from __future__ import annotations

import base64
import csv
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "songs.csv"
CACHE_PATH = Path(os.environ.get("LINK_CACHE", ROOT / "data" / ".link_cache.json"))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124 Safari/537.36")
# Words that mark a different recording of the song; only acceptable when the catalog title has them.
VARIANT_WORDS = ("live", "cover", "karaoke", "remix", "reaction", "slowed", "sped up", "8d", "nightcore",
                 "tutorial", "lesson", "instrumental", "acoustic version", "reverb", "mashup", "piano version")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)           # drop "(Remastered)", "[Official Video]"
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def artist_names(artist: str) -> list[str]:
    parts = re.split(r"\s*(?:,|&|\band\b|\bfeat\.?\b|\bft\.?\b|\bwith\b|/|\bx\b)\s*", artist, flags=re.I)
    names = {norm(artist)} | {norm(p) for p in parts}
    return [n for n in names if len(n) > 1]


def load_rows() -> tuple[list[str], list[str], list[list[str]]]:
    lines = CSV_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    comments = [line for line in lines if line.startswith("#")]
    body = [line for line in lines if not line.startswith("#")]
    reader = list(csv.reader(body))
    return comments, reader[0], reader[1:]


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {"youtube": {}, "spotify": {}}


def save_cache(cache: dict, platform: str) -> None:
    """Merge only this run's platform into the file, so a parallel YouTube and Spotify run never erase each other."""
    latest = load_cache()
    latest[platform].update(cache[platform])
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(latest, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def fetch(url: str, headers: dict | None = None, data: bytes | None = None, retries: int = 4) -> bytes:
    for attempt in range(retries):
        request = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 429:
                wait = int(error.headers.get("Retry-After", "5")) + 1
                print(f"  rate limited, waiting {wait}s", flush=True)
                time.sleep(min(wait, 600))
            elif error.code >= 500:
                time.sleep(2 ** attempt)
            else:
                raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}")


# ---------------------------------------------------------------- YouTube

def youtube_candidates(query: str) -> list[dict]:
    page = fetch("https://www.youtube.com/results?search_query=" + urllib.parse.quote(query),
                 {"Accept-Language": "en-US,en;q=0.9", "Cookie": "CONSENT=YES+1"}).decode("utf-8", "ignore")
    match = re.search(r"var ytInitialData = (\{.*?\});</script>", page, re.S)
    if not match:
        return []
    found: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if "videoRenderer" in node:
                video = node["videoRenderer"]
                try:
                    found.append({
                        "id": video["videoId"],
                        "title": video["title"]["runs"][0]["text"],
                        "channel": video.get("ownerText", {}).get("runs", [{}])[0].get("text", ""),
                    })
                except (KeyError, IndexError):
                    pass
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(match.group(1)))
    return found


def pick_youtube(title: str, artist: str, candidates: list[dict]) -> str:
    want, names = norm(title), artist_names(artist)
    best, best_score = "", 0
    for rank, video in enumerate(candidates[:8]):
        vt, ch = norm(video["title"]), norm(video["channel"])
        if want not in vt:
            continue
        if not any(n in vt or n in ch or ch in n for n in names if n) and not any(n in ch for n in names):
            continue
        raw = video["title"].lower()
        if any(re.search(rf"\b{re.escape(w)}\b", raw) and not re.search(rf"\b{re.escape(w)}\b", title.lower())
               for w in VARIANT_WORDS):
            continue
        score = 10 - rank
        if video["channel"].endswith("- Topic"):
            score += 6
        elif "vevo" in ch or any(n == ch for n in names):
            score += 5
        elif any(n in ch for n in names):
            score += 3
        if "official" in raw:
            score += 2
        if score > best_score:
            best, best_score = video["id"], score
    return f"https://www.youtube.com/watch?v={best}" if best else ""


def run_youtube() -> None:
    _, header, rows = load_rows()
    ti, ai = header.index("title"), header.index("artist")
    cache = load_cache()
    todo = [r for r in rows if r[0] not in cache["youtube"]]
    print(f"YouTube: {len(todo)} songs to look up ({len(rows) - len(todo)} cached)", flush=True)
    for n, row in enumerate(todo, 1):
        sid, title, artist = row[0], row[ti], row[ai]
        try:
            url = pick_youtube(title, artist, youtube_candidates(f"{title} {artist}"))
            if not url:  # one retry biased to the studio upload
                url = pick_youtube(title, artist, youtube_candidates(f"{artist} - {title} official audio"))
        except Exception as error:  # keep going; leave uncached so a rerun retries it
            print(f"  ! {sid} {title}: {error}", flush=True)
            continue
        cache["youtube"][sid] = url
        if n % 10 == 0:
            save_cache(cache, "youtube")
            print(f"  {n}/{len(todo)} (last: {title} -> {url or 'not found'})", flush=True)
        time.sleep(0.6)
    save_cache(cache, "youtube")
    found = sum(1 for v in cache["youtube"].values() if v)
    print(f"YouTube done: {found} found, {len(cache['youtube']) - found} not found", flush=True)


# ---------------------------------------------------------------- Spotify

def spotify_token() -> str:
    cid = os.environ.get("SPOTIFY_CLIENT_ID", "").strip().strip("\"'")
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip().strip("\"'")
    if not cid or not secret:
        sys.exit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET first.")
    # Spotify client IDs and secrets are both 32 hex characters; lengths are safe to print.
    if len(cid) != 32 or len(secret) != 32:
        sys.exit(f"Client ID is {len(cid)} characters and the secret is {len(secret)}; both should be 32. "
                 "Copy them again from the app's Settings page (the secret is hidden until you click "
                 "'View client secret').")
    if cid == secret:
        sys.exit("The Client ID and the secret are identical. Copy the secret separately ('View client secret').")
    for label, value in (("Client ID", cid), ("Client secret", secret)):
        if not re.fullmatch(r"[0-9a-f]{32}", value):
            sys.exit(f"The {label} should only contain digits and lowercase letters a-f; "
                     f"yours has other characters, so it isn't a real Spotify value. Copy it again.")
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    try:
        body = fetch("https://accounts.spotify.com/api/token",
                     {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"},
                     b"grant_type=client_credentials")
    except urllib.error.HTTPError as error:
        sys.exit(f"Spotify rejected the credentials (HTTP {error.code}): {error.read().decode('utf-8', 'ignore')}")
    return json.loads(body)["access_token"]


def pick_spotify(title: str, artist: str, items: list[dict]) -> str:
    want, names = norm(title), artist_names(artist)
    best, best_score = "", 0
    for rank, item in enumerate(items):
        name = norm(item["name"])
        credited = {norm(a["name"]) for a in item["artists"]}
        if not (name == want or name.startswith(want) or want.startswith(name)):
            continue
        if not any(n in credited or any(n in c or c in n for c in credited) for n in names):
            continue
        raw = item["name"].lower()
        if any(re.search(rf"\b{re.escape(w)}\b", raw) and not re.search(rf"\b{re.escape(w)}\b", title.lower())
               for w in VARIANT_WORDS):
            continue
        album = item.get("album", {}).get("album_type", "")
        score = 10 - rank + (5 if name == want else 0) + (1 if album == "album" else 0)
        score += item.get("popularity", 0) / 50
        if score > best_score:
            best, best_score = item["external_urls"]["spotify"], score
    return best


def run_spotify() -> None:
    _, header, rows = load_rows()
    ti, ai = header.index("title"), header.index("artist")
    cache = load_cache()
    todo = [r for r in rows if r[0] not in cache["spotify"]]
    print(f"Spotify: {len(todo)} songs to look up ({len(rows) - len(todo)} cached)", flush=True)
    token = spotify_token()
    for n, row in enumerate(todo, 1):
        sid, title, artist = row[0], row[ti], row[ai]
        first = artist_names(artist)[-1] if artist_names(artist) else artist
        queries = [f'track:"{title}" artist:"{first}"', f"{title} {artist}"]
        url = ""
        try:
            for q in queries:
                params = urllib.parse.urlencode({"q": q, "type": "track", "limit": 10})
                try:
                    data = json.loads(fetch("https://api.spotify.com/v1/search?" + params,
                                            {"Authorization": f"Bearer {token}"}))
                except urllib.error.HTTPError as error:
                    if error.code == 401:
                        token = spotify_token()
                        continue
                    raise
                url = pick_spotify(title, artist, data["tracks"]["items"])
                if url:
                    break
        except Exception as error:
            print(f"  ! {sid} {title}: {error}", flush=True)
            continue
        cache["spotify"][sid] = url
        if n % 25 == 0:
            save_cache(cache, "spotify")
            print(f"  {n}/{len(todo)} (last: {title} -> {url or 'not found'})", flush=True)
        time.sleep(0.15)
    save_cache(cache, "spotify")
    found = sum(1 for v in cache["spotify"].values() if v)
    print(f"Spotify done: {found} found, {len(cache['spotify']) - found} not found", flush=True)


# ---------------------------------------------------------------- merge

def write_csv() -> None:
    comments, header, rows = load_rows()
    cache = load_cache()
    header = [h for h in header if h not in ("youtube_url", "spotify_url")]
    width = len(header)
    out = []
    for row in rows:
        row = row[:width]
        out.append(row + [cache["youtube"].get(row[0], ""), cache["spotify"].get(row[0], "")])
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        handle.writelines(comments)
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header + ["youtube_url", "spotify_url"])
        writer.writerows(out)
    ti, ai = header.index("title"), header.index("artist")
    missing = [(r[0], r[ti], r[ai], not cache["youtube"].get(r[0]), not cache["spotify"].get(r[0])) for r in rows]
    missing = [m for m in missing if m[3] or m[4]]
    report = ROOT / "data" / "links_not_found.tsv"
    report.write_text("id\ttitle\tartist\tmissing\n" + "\n".join(
        f"{i}\t{t}\t{a}\t{'+'.join(p for p, flag in (('youtube', y), ('spotify', s)) if flag)}"
        for i, t, a, y, s in missing) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} rows; {len(missing)} songs missing at least one link -> {report}")


if __name__ == "__main__":
    commands = {"youtube": run_youtube, "spotify": run_spotify, "write": write_csv}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]]()
