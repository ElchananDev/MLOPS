#!/usr/bin/env python3
"""הפקת קובצי שמע לפודקאסט באמצעות Google Cloud Text-to-Speech.

הנגן שב-index.html מקריא במנוע הדיבור של הדפדפן ולא דורש כלום.
הסקריפט הזה נועד למי שרוצה קובצי אודיו אמיתיים להורדה או להפצה.

דרוש מפתח API של Google Cloud Text-to-Speech:

    export GOOGLE_TTS_API_KEY="..."
    python3 make_audio.py --list-voices      # אילו קולות עבריים זמינים
    python3 make_audio.py --dry-run          # כמה תווים וכמה זה יעלה, בלי לקרוא ל-API
    python3 make_audio.py --voice he-IL-Wavenet-D

התוצאה: קובץ WAV לכל פרק בתיקיית audio/, ואם מותקן lame או ffmpeg — גם MP3.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://texttospeech.googleapis.com/v1"
MAX_CHARS = 4500          # מגבלת הבקשה של Google היא 5000 בתים
GAP_SECONDS = 0.45        # שתיקה בין מקטעים


def load_pronunciation() -> list[tuple[str, str]]:
    raw = json.loads((ROOT / "pronounce.json").read_text(encoding="utf-8"))
    return [(item["term"], item["say"]) for item in raw]


PREFIX = r"(^|\s)(וה|מה|כש|שה|לכ|[הובלמכשד])-(?=[\u05d0-\u05ea])"


def normalize_for_speech(text: str) -> str:
    """אחרי התעתיק, מקף בין אותיות עבריות רק גורם למנוע לעצור באמצע מילה."""
    text = text.replace("\u05f3", "'")
    text = re.sub(PREFIX, r"\1\2", text)
    text = re.sub(r"([\u05d0-\u05ea])-(?=[\u05d0-\u05ea])", r"\1 ", text)
    return re.sub(r" {2,}", " ", text)


def to_spoken(text: str, table: list[tuple[str, str]]) -> str:
    """ממיר מונחים באנגלית לכתיב עברי, כדי שקול עברי יהגה אותם נכון."""
    for term, say in table:
        text = re.sub(rf"(^|[^A-Za-z]){re.escape(term)}(?![A-Za-z])", rf"\1{say}", text)
    return normalize_for_speech(text)


def split_for_api(text: str) -> list[str]:
    """מפצל טקסט ארוך למשפטים שלמים שנכנסים במגבלת הבקשה."""
    if len(text.encode()) <= MAX_CHARS:
        return [text]

    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?:…])\s+", text):
        candidate = f"{current} {sentence}".strip()
        if len(candidate.encode()) > MAX_CHARS and current:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def call_api(path: str, key: str, payload: dict | None = None) -> dict:
    url = f"{API}/{path}{'&' if '?' in path else '?'}key={key}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        sys.exit(f"שגיאה מ-Google ({error.code}): {error.read().decode()[:400]}")


def synthesize(text: str, key: str, voice: str, rate: float) -> bytes:
    body = {
        "input": {"text": text},
        "voice": {"languageCode": "he-IL", "name": voice},
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 24000,
            "speakingRate": rate,
        },
    }
    return base64.b64decode(call_api("text:synthesize", key, body)["audioContent"])


def write_episode(chunks: list[bytes], target: Path) -> None:
    """מחבר את מקטעי ה-WAV לקובץ אחד, עם שתיקה קצרה ביניהם."""
    with wave.open(io.BytesIO(chunks[0])) as probe:
        params = probe.getparams()

    silence = b"\x00" * int(params.framerate * params.sampwidth *
                            params.nchannels * GAP_SECONDS)

    with wave.open(str(target), "wb") as out:
        out.setparams(params)
        for index, blob in enumerate(chunks):
            with wave.open(io.BytesIO(blob)) as part:
                out.writeframes(part.readframes(part.getnframes()))
            if index < len(chunks) - 1:
                out.writeframes(silence)


def to_mp3(source: Path) -> Path | None:
    target = source.with_suffix(".mp3")
    if shutil.which("lame"):
        command = ["lame", "--quiet", "-b", "96", str(source), str(target)]
    elif shutil.which("ffmpeg"):
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-i", str(source), "-b:a", "96k", str(target)]
    else:
        return None
    subprocess.run(command, check=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="he-IL-Wavenet-D",
                        help="שם הקול. הרץ עם --list-voices כדי לראות מה זמין")
    parser.add_argument("--rate", type=float, default=1.0, help="קצב דיבור (0.25 עד 4.0)")
    parser.add_argument("--episode", type=int, help="להפיק פרק אחד בלבד")
    parser.add_argument("--out", default="audio", help="תיקיית היעד")
    parser.add_argument("--dry-run", action="store_true",
                        help="ספירת תווים והערכת עלות, בלי לקרוא ל-API")
    parser.add_argument("--list-voices", action="store_true",
                        help="הצגת הקולות העבריים שהחשבון שלך יכול להשתמש בהם")
    args = parser.parse_args()

    key = os.environ.get("GOOGLE_TTS_API_KEY", "")
    if args.list_voices:
        if not key:
            sys.exit("חסר GOOGLE_TTS_API_KEY")
        for item in call_api("voices?languageCode=he-IL", key).get("voices", []):
            print(f"{item['name']:<28} {item.get('ssmlGender','')}")
        return

    table = load_pronunciation()
    episodes = json.loads((ROOT / "script.json").read_text(encoding="utf-8"))
    if args.episode:
        episodes = [e for e in episodes if e["n"] == args.episode]
        if not episodes:
            sys.exit(f"לא נמצא פרק {args.episode}")

    if args.dry_run:
        grand_total = 0
        for episode in episodes:
            chars = requests = 0
            for _, text in episode["segments"]:
                for piece in split_for_api(to_spoken(text, table)):
                    chars += len(piece)
                    requests += 1
            grand_total += chars
            print(f"פרק {episode['n']}: {requests:>3} בקשות, {chars:>6,} תווים — {episode['title']}")
        print(f"\nסך הכול: {grand_total:,} תווים.")
        print("נכון לכתיבת המדריך, המכסה החינמית של Google ל-WaveNet היא מיליון תווים בחודש,")
        print("כלומר הפקה אחת מלאה נכנסת בה בנוחות. ודא את התמחור הנוכחי לפני הפקה חוזרת.")
        return

    if not key:
        sys.exit("חסר GOOGLE_TTS_API_KEY. הרץ עם --dry-run כדי לראות היקף בלי מפתח.")

    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)

    for episode in episodes:
        print(f"פרק {episode['n']}: {episode['title']}")
        chunks: list[bytes] = []
        for index, (_, text) in enumerate(episode["segments"], start=1):
            for piece in split_for_api(to_spoken(text, table)):
                chunks.append(synthesize(piece, key, args.voice, args.rate))
            print(f"  מקטע {index}/{len(episode['segments'])}", end="\r", flush=True)

        wav = out_dir / f"episode-{episode['n']:02d}.wav"
        write_episode(chunks, wav)
        mp3 = to_mp3(wav)
        print(f"  נשמר: {mp3 or wav}" + (" " * 20))

    print("\nהושלם. הקבצים נמצאים ב-" + str(out_dir))


if __name__ == "__main__":
    main()
