import json
import os
import re

# True = skip Whisper, read "vid10 text file" and build chunks (fast).
# False = run Whisper transcribe, then build chunks (slow).
USE_VID10_FILE = True  # True = use vid10 file (fast); False = run Whisper (slow)

AUDIO_PATH = "./audios/10_Video, Audio & Media in HTML.mp3"
VID10_FILE = os.path.join("audios", "vid10 text file")
CHUNKS_OUT = os.path.join("audios", "vid10_chunks.json")


def build_chunks_from_raw_file(path):
    """Parse vid10-style raw string; return list of {start, end, text}. No 'segment' variable."""
    chunks = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        return chunks
    # Match 'start': X, 'end': Y, 'text': '...' (text until ', 'tokens')
    pat = re.compile(
        r"'start':\s*([\d.]+)\s*,\s*'end':\s*([\d.]+)\s*,\s*'text':\s*'((?:(?!',\s*'tokens').)*?)'",
        re.DOTALL,
    )
    for m in pat.finditer(raw):
        try:
            s = float(m.group(1))
            e = float(m.group(2))
            t = m.group(3).strip()
            if t or s != e:
                chunks.append({"start": s, "end": e, "text": t})
        except Exception:
            continue
    return chunks


def build_chunks_from_result(res):
    """Build chunks from Whisper result dict. No 'segment' variable."""
    chunks = []
    raw_list = res.get("segments")
    if not raw_list:
        return chunks
    for item in raw_list:
        try:
            s = item.get("start")
            e = item.get("end")
            t = item.get("text")
            if s is not None and e is not None and t is not None:
                chunks.append({
                    "start": float(s),
                    "end": float(e),
                    "text": str(t).strip(),
                })
        except Exception:
            continue
    return chunks


if USE_VID10_FILE:
    chunks = build_chunks_from_raw_file(VID10_FILE)
else:
    import whisper

    model = whisper.load_model("large")
    result = model.transcribe(
        audio=AUDIO_PATH,
        language="hi",
        task="translate",
        word_timestamps=False,
    )
    chunks = build_chunks_from_result(result)

os.makedirs("audios", exist_ok=True)
with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Saved {len(chunks)} chunks to {CHUNKS_OUT}")
