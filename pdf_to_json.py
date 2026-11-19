"""
One-time converter: Gita chapter PDFs -> gita_chapters/chapter_N.json

Usage:
    python3 pdf_to_json.py <folder_with_pdfs>

Requires the `pdftotext` command (poppler-utils), or `pip install pypdf` as a fallback.
Only the English translation and the commentary are kept: the Sanskrit text is not
used for search (the embedding model is English-only), and PDF extraction garbles
Devanagari vowel signs anyway.
"""
import glob
import json
import os
import re
import subprocess
import sys

OUTPUT_FOLDER = "./gita_chapters"

# Marks the start of every verse block. Tolerates the odd layouts in the PDFs:
#   "Chapter 8, Verse 1\nSanskrit Text:"
#   "...tail of previous verse. Chapter 12, Verse 3 Sanskrit Text:"
#   "Verse 12 Sanskrit Text:"
ANCHOR = re.compile(
    r"(?:Chapter\s+\d+\s*,\s*)?Verse\s+(\d+)\s+Sanskrit(?:\s+Text)?\s*:?",
    re.IGNORECASE,
)
TRANSLATION = re.compile(r"English\s+(?:Translation|Text)\s*:\s*(.*?)(?=Vishesha\s+Vivechana|\Z)", re.S | re.I)
COMMENTARY = re.compile(r"Vishesha\s+Vivechana[^:\n]*:\s*(.*)\Z", re.S | re.I)
# Stray end-of-chapter notes, e.g. "Chapter 8, Verse 29 to Verse 700 ..."
TRAILING_NOTE = re.compile(
    r"(?:Chapter\s+\d+\s*,\s*Verse\s+\d+\s+to\s+Verse\s+\d+"   # "Chapter 8, Verse 29 to Verse 700 ..."
    r"|Closing\s+Editorial\s+Note"                                  # end-of-book note after 18:70
    r"|\u0950[^\n]*\n)"                                              # Devanagari colophon starting with Om
    r".*\Z",
    re.S | re.I,
)


def pdf_to_text(path):
    try:
        return subprocess.run(
            ["pdftotext", "-layout", path, "-"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


# A few Sanskrit words appear inline in the English text with broken vowel signs.
# Map the known ones to a transliteration, then drop any other leftover Devanagari.
INLINE_FIXES = {
    "sevि\u25ccत्वम": "sevitvam",
    "साित्त्वकी": "Sattvic",
    "साित्त्वक": "Sattvic",
    "राजसी": "Rajasic",
    "तामसी": "Tamasic",
}
DEVANAGARI = re.compile(r"[\u0900-\u097F\u25CC]+")


def clean(text):
    for broken, fixed in INLINE_FIXES.items():
        text = text.replace(broken, fixed)
    text = DEVANAGARI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def chapter_number_from_name(path):
    m = re.search(r"chapter[_\s-]*(\d+)", os.path.basename(path), re.I)
    return int(m.group(1)) if m else None


def parse_chapter(text, chapter):
    title_match = re.search(r"CHAPTER\s+\d+\s*:\s*([^\n]+)", text, re.I)
    theme = clean(title_match.group(1)).title() if title_match else f"Chapter {chapter}"

    anchors = list(ANCHOR.finditer(text))
    entries = []
    seen = set()
    last = 0

    for i, a in enumerate(anchors):
        end = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
        block = text[a.end():end]

        t = TRANSLATION.search(block)
        if not t:
            continue
        translation = clean(TRAILING_NOTE.sub("", t.group(1)))

        # The previous verse's trailing text can be glued to the next header on the
        # same line ("... welfare of the world. Chapter 12, Verse 1"); the ANCHOR regex
        # already consumed the header, so only a stray "Chapter N," prefix could remain.
        translation = re.sub(r"\s*Chapter\s+\d+\s*,?\s*$", "", translation)

        c = COMMENTARY.search(block)
        commentary = clean(re.sub(r"\s*Chapter\s+\d+\s*,?\s*$", "", TRAILING_NOTE.sub("", c.group(1)))) if c else ""

        verse = int(a.group(1))
        # Some headers are mistyped in the source (e.g. a second "Verse 44" that is
        # really verse 54). A repeated number is renumbered to follow the previous verse.
        if verse in seen:
            verse = last + 1
        seen.add(verse)
        last = verse
        entries.append({
            "id": f"{chapter}_{verse}",
            "chapter": chapter,
            "verse": verse,
            "theme": theme,
            "translation": translation,
            "simple_explanation": commentary,
        })
    return entries


def main(folder):
    pdfs = sorted(glob.glob(os.path.join(folder, "*.pdf")), key=lambda p: chapter_number_from_name(p) or 0)
    if not pdfs:
        sys.exit(f"No PDFs found in {folder}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    total = 0
    for path in pdfs:
        chapter = chapter_number_from_name(path)
        if chapter is None:
            print(f"Skipping (no chapter number in name): {path}")
            continue

        entries = parse_chapter(pdf_to_text(path), chapter)

        # Report gaps/duplicates so bad parses are visible
        nums = [e["verse"] for e in entries]
        missing = sorted(set(range(1, max(nums) + 1)) - set(nums)) if nums else []
        dupes = sorted({n for n in nums if nums.count(n) > 1})
        status = "OK" if not missing and not dupes else f"CHECK missing={missing} dupes={dupes}"
        print(f"Chapter {chapter:>2}: {len(entries):>3} verses  {status}")

        with open(os.path.join(OUTPUT_FOLDER, f"chapter_{chapter}.json"), "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        total += len(entries)

    print(f"\nDone: {total} verses written to {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
