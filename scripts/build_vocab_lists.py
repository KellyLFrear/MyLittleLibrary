#!/usr/bin/env python3
"""
build_vocab_lists.py

Purpose:
- Read raw vocabulary source files from an input folder
- Normalize and merge their rows
- Score each word for Beginner / Intermediate / Advanced suitability
- Build three exclusive internal bands:
    * beginner add-on = 1000 words
    * intermediate add-on = 2000 words
    * advanced add-on = 3000 words
- Write cumulative runtime files:
    * beginner_1000.txt      -> 1000 total words
    * intermediate_3000.txt  -> 3000 total words
    * advanced_6000.txt      -> 6000 total words

Big picture:
This is the script that turns many possible source lists into the final project
vocabulary lists used by analyze_articles.py.

Key ideas used in scoring:
- source type / source tag
- grade hints from input files
- rank and frequency count if present
- academic flag if present
- part of speech hints
- word shape / suffix heuristics
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Accept only single-word lowercase vocabulary entries, with optional internal
# apostrophes or hyphens such as "can't" or "well-known".
WORD_RE = re.compile(r"^[a-z]+(?:['-][a-z]+)*$")

# Final cumulative vocabulary totals required by the project at runtime.
RUNTIME_BEGINNER_TOTAL = 1000
RUNTIME_INTERMEDIATE_TOTAL = 3000
RUNTIME_ADVANCED_TOTAL = 6000

# Internal exclusive band sizes. These are the increments added at each stage.
BEGINNER_ADDON_SIZE = RUNTIME_BEGINNER_TOTAL
INTERMEDIATE_ADDON_SIZE = RUNTIME_INTERMEDIATE_TOTAL - RUNTIME_BEGINNER_TOTAL  # 2000
ADVANCED_ADDON_SIZE = RUNTIME_ADVANCED_TOTAL - RUNTIME_INTERMEDIATE_TOTAL      # 3000

BEGINNER_GRADE_HINT = "k-5th grade"
INTERMEDIATE_GRADE_HINT = "6-8"
ADVANCED_GRADE_HINT = "9-12"

FUNCTION_WORD_POS = {
    "article", "determiner", "pronoun", "conjunction", "preposition", "auxiliary"
}

ACADEMIC_SUFFIXES = (
    "tion", "sion", "ment", "ity", "ence", "ance", "ology",
    "ism", "ist", "ous", "ive", "ate", "ize", "ality", "ship"
)

CANONICAL_RUNTIME_TOTALS = {
    "beginner": RUNTIME_BEGINNER_TOTAL,
    "intermediate": RUNTIME_INTERMEDIATE_TOTAL,
    "advanced": RUNTIME_ADVANCED_TOTAL,
}

EXCLUSIVE_ADDON_SIZES = {
    "beginner": BEGINNER_ADDON_SIZE,
    "intermediate": INTERMEDIATE_ADDON_SIZE,
    "advanced": ADVANCED_ADDON_SIZE,
}

EXPLICIT_GRADE_HINT_ALIASES = {
    # Beginner
    "beginner": BEGINNER_GRADE_HINT,
    "k": BEGINNER_GRADE_HINT,
    "0": BEGINNER_GRADE_HINT,
    "kindergarten": BEGINNER_GRADE_HINT,
    "pre-k": BEGINNER_GRADE_HINT,
    "pre k": BEGINNER_GRADE_HINT,
    "prek": BEGINNER_GRADE_HINT,
    "k-4": BEGINNER_GRADE_HINT,
    "k-4th": BEGINNER_GRADE_HINT,
    "k-4 grade": BEGINNER_GRADE_HINT,
    "k-4th grade": BEGINNER_GRADE_HINT,
    "k to 4": BEGINNER_GRADE_HINT,
    "k to 4th": BEGINNER_GRADE_HINT,
    "kindergarten to 4th grade": BEGINNER_GRADE_HINT,
    "kindergarten-4th grade": BEGINNER_GRADE_HINT,
    "k-5": BEGINNER_GRADE_HINT,
    "k-5th": BEGINNER_GRADE_HINT,
    "k-5 grade": BEGINNER_GRADE_HINT,
    "k-5th grade": BEGINNER_GRADE_HINT,
    "k to 5": BEGINNER_GRADE_HINT,
    "k to 5th": BEGINNER_GRADE_HINT,
    "kindergarten to 5th grade": BEGINNER_GRADE_HINT,
    "kindergarten-5th grade": BEGINNER_GRADE_HINT,
    "1-5": BEGINNER_GRADE_HINT,
    # Intermediate
    "intermediate": INTERMEDIATE_GRADE_HINT,
    "6-8": INTERMEDIATE_GRADE_HINT,
    "6th-8th grade": INTERMEDIATE_GRADE_HINT,
    "6-8th grade": INTERMEDIATE_GRADE_HINT,
    "6 to 8": INTERMEDIATE_GRADE_HINT,
    "6 to 8th": INTERMEDIATE_GRADE_HINT,
    # Advanced
    "advanced": ADVANCED_GRADE_HINT,
    "9-12": ADVANCED_GRADE_HINT,
    "9th-12th grade": ADVANCED_GRADE_HINT,
    "9-12th grade": ADVANCED_GRADE_HINT,
    "9 to 12": ADVANCED_GRADE_HINT,
    "9 to 12th": ADVANCED_GRADE_HINT,
}


@dataclass
class SourceRow:
    """
    One row as read from a source file before merging duplicates.

    Example:
    The same word may appear in multiple CSV/TXT files, possibly with different
    metadata. Later we merge those rows into one VocabEntry.
    """
    word: str
    source_file: str
    source_tag: str
    source_list: str
    source_grade_hint: Optional[str] = None
    normalized_band: Optional[str] = None
    rank: Optional[int] = None
    count: Optional[int] = None
    academic: Optional[bool] = None
    pos: Optional[str] = None
    notes: str = ""


@dataclass
class VocabEntry:
    """
    Merged representation of a single vocabulary word across all source files.

    Stores:
    - where the word came from
    - collected metadata
    - scoring information for all three difficulty bands
    """
    word: str
    source_files: Set[str] = field(default_factory=set)
    source_tags: Set[str] = field(default_factory=set)
    source_lists: Set[str] = field(default_factory=set)
    source_grade_hints: Set[str] = field(default_factory=set)
    normalized_bands: Set[str] = field(default_factory=set)
    min_rank: Optional[int] = None
    max_count: Optional[int] = None
    academic_votes_true: int = 0
    academic_votes_false: int = 0
    pos_tags: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)

    beginner_score: float = 0.0
    intermediate_score: float = 0.0
    advanced_score: float = 0.0
    preferred_band: str = ""

    def academic_flag(self) -> Optional[bool]:
        total = self.academic_votes_true + self.academic_votes_false
        if total == 0:
            return None
        return self.academic_votes_true >= self.academic_votes_false

    def best_normalized_band(self) -> Optional[str]:
        if len(self.normalized_bands) == 1:
            return next(iter(self.normalized_bands))
        return None

    def normalized_band_text(self) -> str:
        return ";".join(sorted(self.normalized_bands))

    def source_grade_hint_text(self) -> str:
        return ";".join(sorted(self.source_grade_hints))

    def source_list_text(self) -> str:
        return ";".join(sorted(self.source_lists))

    def primary_pos(self) -> str:
        if not self.pos_tags:
            return ""
        return sorted(self.pos_tags)[0]


def parse_args() -> argparse.Namespace:
    """Read command-line options for input/output locations and build mode."""
    parser = argparse.ArgumentParser(description="Build banded vocabulary lists from source files.")
    parser.add_argument("--input", required=True, help="Input directory containing CSV/TXT source files.")
    parser.add_argument("--output", required=True, help="Output directory for scored CSV and summary files.")
    parser.add_argument(
        "--runtime-vocab-dir",
        default=None,
        help="Optional directory for runtime one-word-per-line TXT files, e.g. data/vocab",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Allow fewer than the target counts if the sources are too small.",
    )
    parser.add_argument(
        "--only-band",
        choices=["all", "beginner", "intermediate", "advanced"],
        default="all",
        help="Build all bands or only one specific runtime band.",
    )
    return parser.parse_args()


def normalize_text(value: Optional[str]) -> str:
    """Lowercase and normalize punctuation/whitespace for general text fields."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_word(raw: str) -> Optional[str]:
    """
    Clean a raw token and return a normalized vocabulary word.

    Rejects:
    - blanks
    - multi-word entries
    - tokens that do not match the allowed word pattern
    """
    word = normalize_text(raw)
    word = re.sub(r"^[^a-z]+|[^a-z]+$", "", word)

    if " " in word:
        return None
    if not word:
        return None
    if not WORD_RE.fullmatch(word):
        return None
    return word


def parse_bool(value: Optional[str]) -> Optional[bool]:
    """Parse common boolean-like strings such as 1/0, true/false, yes/no."""
    if value is None:
        return None
    s = normalize_text(value)
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def parse_int(value: Optional[str]) -> Optional[int]:
    """Parse an integer field safely, allowing values like "12" or "12.0"."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _band_for_single_grade(level: int) -> Optional[str]:
    """Map a single grade number to one of the three project bands."""
    if 0 <= level <= 5:
        return BEGINNER_GRADE_HINT
    if 6 <= level <= 8:
        return INTERMEDIATE_GRADE_HINT
    if 9 <= level <= 12:
        return ADVANCED_GRADE_HINT
    return None


def parse_grade_hint(value: Optional[str]) -> Optional[str]:
    
    """
    Accept flexible source labels and normalize them to one of:
    - k-5th grade
    - 6-8
    - 9-12
    """
    s = normalize_text(value)
    if not s:
        return None

    if s in EXPLICIT_GRADE_HINT_ALIASES:
        return EXPLICIT_GRADE_HINT_ALIASES[s]

    compact = re.sub(r"\b(grades?|grade level|level)\b", "", s)
    compact = compact.replace("kindergarten", "k")
    compact = re.sub(r"(st|nd|rd|th)\b", "", compact)
    compact = re.sub(r"\s+", " ", compact).strip()

    if compact in EXPLICIT_GRADE_HINT_ALIASES:
        return EXPLICIT_GRADE_HINT_ALIASES[compact]

    if compact == "k":
        return BEGINNER_GRADE_HINT
    if compact.isdigit():
        return _band_for_single_grade(int(compact))

    range_match = re.fullmatch(r"(k|\d{1,2})\s*(?:-|to)\s*(\d{1,2})", compact)
    if range_match:
        start_raw, end_raw = range_match.groups()
        start = 0 if start_raw == "k" else int(start_raw)
        end = int(end_raw)
        if start <= 5 and end <= 5:
            return BEGINNER_GRADE_HINT
        if 6 <= start <= 8 and 6 <= end <= 8:
            return INTERMEDIATE_GRADE_HINT
        if 9 <= start <= 12 and 9 <= end <= 12:
            return ADVANCED_GRADE_HINT

    return None


def infer_source_tag(filename: str) -> str:
    """Infer the type of source file from its filename so it can influence scoring."""
    name = filename.lower()

    if "banned" in name or "exclude" in name:
        return "banned"

    if any(k in name for k in [
        "dolch", "fry", "beginner", "elementary", "elem", "k4", "k5",
        "grade1", "grade2", "grade3", "grade4", "grade5",
    ]):
        return "beginner_seed"

    if any(k in name for k in [
        "ngsl", "intermediate", "middle", "grade6", "grade7", "grade8", "ms",
    ]):
        return "intermediate_seed"

    if any(k in name for k in [
        "awl", "advanced", "highschool", "high_school",
        "grade9", "grade10", "grade11", "grade12", "hs",
    ]):
        return "advanced_seed"

    if "academic" in name:
        return "academic_corpus"

    if any(k in name for k in ["coca", "bnc", "corpus", "frequency", "general"]):
        return "general_frequency"

    if any(k in name for k in ["children", "kid", "reader"]):
        return "children_corpus"

    return "unknown"


def load_banned_words(input_dir: Path) -> Set[str]:
    """Load words from files whose names imply they are exclusion/banned lists."""
    banned: Set[str] = set()
    for path in input_dir.iterdir():
        if not path.is_file() or infer_source_tag(path.name) != "banned":
            continue

        if path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                word = normalize_word(line)
                if word:
                    banned.add(word)

        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = normalize_word(row.get("word", ""))
                    if word:
                        banned.add(word)
    return banned


def read_txt_source(path: Path, source_tag: str) -> Iterable[SourceRow]:
    """Read a plain TXT source file where each non-empty line is a word."""
    source_list = path.stem
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        word = normalize_word(stripped)
        if not word:
            continue
        yield SourceRow(
            word=word,
            source_file=path.name,
            source_tag=source_tag,
            source_list=source_list,
        )


def read_csv_source(path: Path, source_tag: str) -> Iterable[SourceRow]:
    """Read a CSV source file and map each valid row into a SourceRow object."""
    source_list_default = path.stem
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "word" not in reader.fieldnames:
            raise ValueError(f"{path.name} must include a 'word' column.")

        for row in reader:
            word = normalize_word(row.get("word", ""))
            if not word:
                continue

            source_grade_hint_raw = (row.get("source_grade_hint") or row.get("grade_hint") or "").strip()
            normalized_band_raw = (row.get("normalized_band") or "").strip()
            normalized_band = parse_grade_hint(normalized_band_raw) or parse_grade_hint(source_grade_hint_raw)

            yield SourceRow(
                word=word,
                source_file=path.name,
                source_tag=(row.get("source_tag") or "").strip() or source_tag,
                source_list=(row.get("source_list") or "").strip() or source_list_default,
                source_grade_hint=source_grade_hint_raw or None,
                normalized_band=normalized_band,
                rank=parse_int(row.get("rank")),
                count=parse_int(row.get("count")),
                academic=parse_bool(row.get("academic")),
                pos=(row.get("pos") or "").strip().lower() or None,
                notes=(row.get("notes") or "").strip(),
            )


def load_rows(input_dir: Path) -> List[SourceRow]:
    """Load all non-banned TXT/CSV source rows from the input directory."""
    rows: List[SourceRow] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue

        source_tag = infer_source_tag(path.name)
        if source_tag == "banned":
            continue

        suffix = path.suffix.lower()
        if suffix == ".txt":
            rows.extend(read_txt_source(path, source_tag))
        elif suffix == ".csv":
            rows.extend(read_csv_source(path, source_tag))
    return rows


def merge_rows(rows: Iterable[SourceRow], banned: Set[str]) -> Dict[str, VocabEntry]:
    """Merge duplicate word rows into one VocabEntry while collecting metadata."""
    merged: Dict[str, VocabEntry] = {}

    for row in rows:
        if row.word in banned:
            continue

        entry = merged.get(row.word)
        if entry is None:
            entry = VocabEntry(word=row.word)
            merged[row.word] = entry

        entry.source_files.add(row.source_file)
        entry.source_tags.add(row.source_tag)
        entry.source_lists.add(row.source_list)

        if row.source_grade_hint:
            entry.source_grade_hints.add(row.source_grade_hint)

        if row.normalized_band:
            entry.normalized_bands.add(row.normalized_band)

        if row.rank is not None:
            entry.min_rank = row.rank if entry.min_rank is None else min(entry.min_rank, row.rank)

        if row.count is not None:
            entry.max_count = row.count if entry.max_count is None else max(entry.max_count, row.count)

        if row.academic is True:
            entry.academic_votes_true += 1
        elif row.academic is False:
            entry.academic_votes_false += 1

        if row.pos:
            entry.pos_tags.add(row.pos)

        if row.notes:
            entry.notes.append(row.notes)

    return merged


def apply_source_tag_scores(entry: VocabEntry) -> None:
    """Score a word based on what kind of source file(s) it came from."""
    for tag in entry.source_tags:
        if tag == "beginner_seed":
            entry.beginner_score += 12
            entry.intermediate_score += 2
        elif tag == "intermediate_seed":
            entry.intermediate_score += 12
            entry.advanced_score += 4
        elif tag == "advanced_seed":
            entry.advanced_score += 12
            entry.intermediate_score += 4
            entry.beginner_score -= 1
        elif tag == "general_frequency":
            entry.beginner_score += 3
            entry.intermediate_score += 5
            entry.advanced_score += 5
        elif tag == "children_corpus":
            entry.beginner_score += 8
            entry.intermediate_score += 3
        elif tag == "academic_corpus":
            entry.intermediate_score += 4
            entry.advanced_score += 10
        elif tag == "unknown":
            entry.intermediate_score += 2
            entry.advanced_score += 2


def apply_rank_scores(entry: VocabEntry) -> None:
    """Boost scores using rank metadata when lower rank implies more common words."""
    rank = entry.min_rank
    if rank is None:
        return

    if rank <= 1000:
        entry.beginner_score += 6
        entry.intermediate_score += 5
        entry.advanced_score += 2
    elif rank <= 3000:
        entry.beginner_score += 2
        entry.intermediate_score += 6
        entry.advanced_score += 4
    elif rank <= 6000:
        entry.intermediate_score += 4
        entry.advanced_score += 6
    elif rank <= 10000:
        entry.intermediate_score += 2
        entry.advanced_score += 5
    else:
        entry.advanced_score += 2


def apply_count_scores(entry: VocabEntry) -> None:
    """Boost scores using frequency count metadata with a log-scaled bonus."""
    count = entry.max_count
    if count is None or count <= 0:
        return

    bonus = math.log10(count + 10)
    entry.beginner_score += min(3.0, bonus * 0.6)
    entry.intermediate_score += min(4.0, bonus * 0.8)
    entry.advanced_score += min(4.0, bonus * 0.7)


def apply_grade_hint_scores(entry: VocabEntry) -> None:
    """Boost the band score suggested by explicit source grade metadata."""
    hint = entry.best_normalized_band()
    if hint == BEGINNER_GRADE_HINT:
        entry.beginner_score += 10
    elif hint == INTERMEDIATE_GRADE_HINT:
        entry.intermediate_score += 10
    elif hint == ADVANCED_GRADE_HINT:
        entry.advanced_score += 10


def apply_academic_scores(entry: VocabEntry) -> None:
    """Adjust scores using the academic/non-academic flag when available."""
    academic = entry.academic_flag()
    if academic is True:
        entry.advanced_score += 4
        entry.intermediate_score += 1
        entry.beginner_score -= 1
    elif academic is False:
        entry.beginner_score += 1


def apply_shape_scores(entry: VocabEntry) -> None:
    """Use simple heuristics such as word length and suffixes to adjust scores."""
    word = entry.word
    length = len(word)

    if length <= 4:
        entry.beginner_score += 1.8
    elif length <= 6:
        entry.beginner_score += 0.8
        entry.intermediate_score += 0.5
    elif length >= 8:
        entry.intermediate_score += 0.5
        entry.advanced_score += 0.7

    if length >= 11:
        entry.advanced_score += 1.5

    if any(word.endswith(suffix) for suffix in ACADEMIC_SUFFIXES):
        entry.intermediate_score += 0.5
        entry.advanced_score += 1.4


def apply_pos_scores(entry: VocabEntry) -> None:
    """Favor beginner scoring for common function-word parts of speech."""
    pos_tags = {p.lower() for p in entry.pos_tags}
    if pos_tags & FUNCTION_WORD_POS:
        entry.beginner_score += 2.0


def choose_preferred_band(entry: VocabEntry) -> str:
    """Choose the final best-fit band, preferring explicit hints when present."""
    hint = entry.best_normalized_band()
    if hint == BEGINNER_GRADE_HINT:
        return "beginner"
    if hint == INTERMEDIATE_GRADE_HINT:
        return "intermediate"
    if hint == ADVANCED_GRADE_HINT:
        return "advanced"

    scores = {
        "beginner": entry.beginner_score,
        "intermediate": entry.intermediate_score,
        "advanced": entry.advanced_score,
    }
    return max(scores, key=scores.get)


def score_entries(entries: Dict[str, VocabEntry]) -> None:
    """Run all scoring rules for every merged vocabulary entry."""
    for entry in entries.values():
        apply_source_tag_scores(entry)
        apply_rank_scores(entry)
        apply_count_scores(entry)
        apply_grade_hint_scores(entry)
        apply_academic_scores(entry)
        apply_shape_scores(entry)
        apply_pos_scores(entry)
        entry.preferred_band = choose_preferred_band(entry)


def sort_for_band(entries: Iterable[VocabEntry], band: str) -> List[VocabEntry]:
    """Sort words for a target band using score, rank, source count, then alphabetically."""
    def key_fn(entry: VocabEntry) -> Tuple[float, int, int, str]:
        score = {
            "beginner": entry.beginner_score,
            "intermediate": entry.intermediate_score,
            "advanced": entry.advanced_score,
        }[band]
        rank = entry.min_rank if entry.min_rank is not None else 10**9
        source_count = len(entry.source_files)
        return (-score, rank, -source_count, entry.word)

    return sorted(entries, key=key_fn)


def take_band(entries: Iterable[VocabEntry], band: str, size: int) -> List[VocabEntry]:
    """Take the top N words for a given band after sorting."""
    return sort_for_band(entries, band)[:size]


def build_exclusive_bands(all_entries_sorted: List[VocabEntry]) -> Dict[str, List[VocabEntry]]:
    """Build non-overlapping beginner, intermediate, and advanced add-on bands."""
    remaining = {entry.word: entry for entry in all_entries_sorted}

    beginner = take_band(remaining.values(), "beginner", BEGINNER_ADDON_SIZE)
    for entry in beginner:
        remaining.pop(entry.word, None)

    intermediate = take_band(remaining.values(), "intermediate", INTERMEDIATE_ADDON_SIZE)
    for entry in intermediate:
        remaining.pop(entry.word, None)

    advanced = take_band(remaining.values(), "advanced", ADVANCED_ADDON_SIZE)

    return {
        "beginner": beginner,
        "intermediate": intermediate,
        "advanced": advanced,
    }


def validate_needed(selection: Dict[str, List[VocabEntry]], requested_band: str, allow_shortfall: bool) -> None:
    """Ensure enough words were found to satisfy the requested target sizes."""
    requirements = {
        "beginner": {"beginner": BEGINNER_ADDON_SIZE},
        "intermediate": {"beginner": BEGINNER_ADDON_SIZE, "intermediate": INTERMEDIATE_ADDON_SIZE},
        "advanced": {"beginner": BEGINNER_ADDON_SIZE, "intermediate": INTERMEDIATE_ADDON_SIZE, "advanced": ADVANCED_ADDON_SIZE},
        "all": {"beginner": BEGINNER_ADDON_SIZE, "intermediate": INTERMEDIATE_ADDON_SIZE, "advanced": ADVANCED_ADDON_SIZE},
    }[requested_band]

    missing = {}
    for band, required_size in requirements.items():
        actual = len(selection.get(band, []))
        if actual < required_size:
            missing[band] = required_size - actual

    if not missing or allow_shortfall:
        return

    missing_text = ", ".join(f"{band}: need {count} more" for band, count in missing.items())
    raise SystemExit(
        "Not enough usable source words to build the requested vocab band(s). "
        f"Shortfall -> {missing_text}. Add more rows to vocab_sources/ or rerun with --allow-shortfall."
    )


def cumulative_runtime_rows(selection: Dict[str, List[VocabEntry]], band: str) -> List[VocabEntry]:
    """Return the cumulative runtime list for a band, including lower bands below it."""
    if band == "beginner":
        return list(selection["beginner"])
    if band == "intermediate":
        return list(selection["beginner"]) + list(selection["intermediate"])
    if band == "advanced":
        return list(selection["beginner"]) + list(selection["intermediate"]) + list(selection["advanced"])
    raise ValueError(f"Unknown band: {band}")


def entry_to_row(entry: VocabEntry, band: Optional[str] = None) -> Dict[str, str]:
    """Convert a VocabEntry into a CSV-friendly dictionary row."""
    return {
        "word": entry.word,
        "source_grade_hint": entry.source_grade_hint_text(),
        "normalized_band": entry.normalized_band_text(),
        "source_list": entry.source_list_text(),
        "rank": entry.min_rank if entry.min_rank is not None else "",
        "count": entry.max_count if entry.max_count is not None else "",
        "academic": "" if entry.academic_flag() is None else str(entry.academic_flag()).lower(),
        "pos": entry.primary_pos(),
        "notes": " | ".join(dict.fromkeys(entry.notes)),
        "band": band or "",
        "preferred_band": entry.preferred_band,
        "beginner_score": round(entry.beginner_score, 3),
        "intermediate_score": round(entry.intermediate_score, 3),
        "advanced_score": round(entry.advanced_score, 3),
        "source_tags": ";".join(sorted(entry.source_tags)),
        "source_files": ";".join(sorted(entry.source_files)),
    }


def write_csv(path: Path, rows: List[VocabEntry], band: str) -> None:
    """Write one exclusive band to a CSV file with metadata and scores."""
    fieldnames = [
        "word",
        "source_grade_hint",
        "normalized_band",
        "source_list",
        "rank",
        "count",
        "academic",
        "pos",
        "notes",
        "band",
        "preferred_band",
        "beginner_score",
        "intermediate_score",
        "advanced_score",
        "source_tags",
        "source_files",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in rows:
            writer.writerow(entry_to_row(entry, band=band))


def write_master_csv(path: Path, rows: List[VocabEntry]) -> None:
    """Write a CSV containing every merged/scored vocabulary entry."""
    fieldnames = [
        "word",
        "source_grade_hint",
        "normalized_band",
        "source_list",
        "rank",
        "count",
        "academic",
        "pos",
        "notes",
        "preferred_band",
        "beginner_score",
        "intermediate_score",
        "advanced_score",
        "source_tags",
        "source_files",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in rows:
            row = entry_to_row(entry)
            row.pop("band", None)
            writer.writerow(row)


def write_txt(path: Path, rows: List[VocabEntry]) -> None:
    """Write a simple one-word-per-line TXT file for runtime use."""
    unique_words = sorted({entry.word for entry in rows})
    with path.open("w", encoding="utf-8") as f:
        for word in unique_words:
            f.write(f"{word}\n")


def write_runtime_file(runtime_dir: Path, band: str, rows: List[VocabEntry]) -> str:
    """Write the final runtime TXT file for one cumulative band and return its path."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    name_map = {
        "beginner": "beginner_1000.txt",
        "intermediate": "intermediate_3000.txt",
        "advanced": "advanced_6000.txt",
    }
    path = runtime_dir / name_map[band]
    write_txt(path, rows)
    return str(path)


def main() -> None:
    """Run the full vocabulary-build pipeline from source files to output artifacts."""
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    runtime_dir = Path(args.runtime_vocab_dir) if args.runtime_vocab_dir else output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # Load exclusions, read all usable source rows, merge duplicates,
    # then compute Beginner/Intermediate/Advanced scores for each word.
    banned = load_banned_words(input_dir)
    rows = load_rows(input_dir)
    merged = merge_rows(rows, banned)
    score_entries(merged)

    # Build exclusive add-on bands. A word chosen for beginner is removed
    # before intermediate selection, and so on.
    all_entries_sorted = sorted(merged.values(), key=lambda entry: entry.word)
    exclusive = build_exclusive_bands(all_entries_sorted)

    validate_needed(exclusive, args.only_band, args.allow_shortfall)

    output_files: Dict[str, str] = {}
    runtime_counts: Dict[str, int] = {}
    exclusive_counts = {band: len(rows) for band, rows in exclusive.items()}

    # Always write master CSV.
    master_csv = output_dir / "master_vocab_scored.csv"
    write_master_csv(master_csv, all_entries_sorted)
    output_files["master_csv"] = str(master_csv)

    # Write exclusive band CSVs.
    csv_specs = {
        "beginner": ("beginner_band_1000.csv", BEGINNER_ADDON_SIZE),
        "intermediate": ("intermediate_band_2000.csv", INTERMEDIATE_ADDON_SIZE),
        "advanced": ("advanced_band_3000.csv", ADVANCED_ADDON_SIZE),
    }
    if args.only_band == "all":
        csv_bands = ["beginner", "intermediate", "advanced"]
    else:
        csv_bands = [args.only_band]

    for band in csv_bands:
        filename, _ = csv_specs[band]
        csv_path = output_dir / filename
        write_csv(csv_path, exclusive[band], band)
        output_files[f"{band}_band_csv"] = str(csv_path)

    # Write runtime cumulative TXT files.
    if args.only_band == "all":
        runtime_bands = ["beginner", "intermediate", "advanced"]
    else:
        runtime_bands = [args.only_band]

    for band in runtime_bands:
        runtime_rows = cumulative_runtime_rows(exclusive, band)
        runtime_counts[band] = len({entry.word for entry in runtime_rows})
        output_files[f"{band}_txt"] = write_runtime_file(runtime_dir, band, runtime_rows)

    # Record a machine-readable summary so you can verify counts and output paths.
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "runtime_vocab_dir": str(runtime_dir),
        "only_band": args.only_band,
        "banned_words_count": len(banned),
        "source_rows_loaded": len(rows),
        "unique_words_after_merge": len(all_entries_sorted),
        "exclusive_band_counts": exclusive_counts,
        "runtime_total_counts": runtime_counts,
        "expected_runtime_totals": (
            CANONICAL_RUNTIME_TOTALS if args.only_band == "all"
            else {args.only_band: CANONICAL_RUNTIME_TOTALS[args.only_band]}
        ),
        "output_files": output_files,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
