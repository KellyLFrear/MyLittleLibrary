from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

from src.embeddings.chunker import ArticleChunk
from src.rag.retriever import TwoStageRetriever

if TYPE_CHECKING:
    from src.rag.student_profile import StudentProfile


# ── Story vocabulary analysis helpers ─────────────────────────────────────────

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


STORY_TARGET_KNOWN_RANGES: Dict[str, Tuple[float, float]] = {
    "low": (0.95, 0.98),
    "medium": (0.90, 0.95),
    "high": (0.85, 0.92),
}

# Allows near-misses like 0.899 to count as acceptable for a 0.90 lower bound.
STORY_TARGET_TOLERANCE = 0.01


BASIC_CHALLENGE_WORD_BLOCKLIST: set[str] = {
    # Very common/basic words that should not be presented as challenge vocabulary.
    "happy", "sad", "mad", "glad", "good", "bad", "big", "small", "little",
    "nice", "kind", "fun", "new", "old", "young", "long", "short",

    # Common concrete nouns/verbs that are not useful as explicit vocab targets.
    "lake", "tree", "trees", "flower", "flowers", "rock", "sand", "planet",
    "space", "star", "stars", "hand", "hands", "face", "faces",

    # Common inflected/simple action words.
    "met", "got", "came", "went", "made", "took", "gave", "saw", "looked",
    "started", "watched", "moved", "danced", "clapped", "cheered", "walked",
    "landed", "searched", "received", "prepared", "formed", "grew",

    # Common school-level words that usually do not need explicit teaching here.
    "crew", "message", "center", "middle", "empty", "quiet", "fear",
}


def _normalize_word(word: str) -> str:
    """Normalize a word for vocabulary matching."""
    word = word.strip().lower()
    if word.endswith("'s"):
        word = word[:-2]
    return word.strip("'")


def _candidate_word_forms(word: str) -> set[str]:
    """
    Return simple word-family candidates for vocabulary matching.

    Examples:
    - flowers  -> flower
    - cheered  -> cheer
    - clapped  -> clap
    - bringing -> bring
    - studies  -> study
    - danced   -> dance
    - excited  -> excite

    This is intentionally lightweight. It is not a full stemmer, but it fixes
    the most common inflection cases without adding another dependency.
    """
    word = _normalize_word(word)
    if not word:
        return set()

    forms = {word}

    # Plurals: stories -> story
    if len(word) > 4 and word.endswith("ies"):
        forms.add(word[:-3] + "y")

    # Plurals: flowers -> flower, planets -> planet
    if len(word) > 3 and word.endswith("s"):
        forms.add(word[:-1])

    # Plurals / variants: boxes -> box, buses -> bus
    if len(word) > 4 and word.endswith("es"):
        forms.add(word[:-2])

    # Past tense: cheered -> cheer, bloomed -> bloom, danced -> dance, excited -> excite
    if len(word) > 4 and word.endswith("ed"):
        base = word[:-2]
        forms.add(base)

        # Double consonant: clapped -> clap, stopped -> stop
        if len(base) > 2 and base[-1] == base[-2]:
            forms.add(base[:-1])

        # Silent-e recovery: danced -> dance, excited -> excite
        forms.add(base + "e")

    # Gerund: bringing -> bring, running -> run, making -> make
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        forms.add(base)

        # Double consonant: running -> run
        if len(base) > 2 and base[-1] == base[-2]:
            forms.add(base[:-1])

        # Silent-e recovery: making -> make
        forms.add(base + "e")

    # Comparative/superlative: brighter -> bright, fastest -> fast
    if len(word) > 4 and word.endswith("er"):
        forms.add(word[:-2])
    if len(word) > 5 and word.endswith("est"):
        forms.add(word[:-3])

    # Adverbs: slowly -> slow
    if len(word) > 4 and word.endswith("ly"):
        forms.add(word[:-2])

    return {form for form in forms if form}


def _is_known_word(token: str, known_set: set[str]) -> bool:
    """Return True if token or a simple word-family form is known."""
    return any(form in known_set for form in _candidate_word_forms(token))


def _challenge_word_score(word: str) -> int:
    """
    Score whether an unknown word is useful as an explicit challenge word.

    Higher score = better word to show in new_vocab.
    This does not change coverage math; it only improves the displayed vocab list.
    """
    word = _normalize_word(word)

    if not word:
        return -100

    if word in BASIC_CHALLENGE_WORD_BLOCKLIST:
        return -100

    # Very short words are usually not useful as challenge vocabulary.
    if len(word) <= 3:
        return -50

    score = 0

    # Longer words are often better learning targets.
    if len(word) >= 6:
        score += 2
    if len(word) >= 9:
        score += 2

    # Academic/science/story words often carry useful meaning.
    useful_suffixes = (
        "tion", "sion", "ment", "ness", "ity", "ous", "ive", "al",
        "ic", "ize", "izing", "less", "ful",
    )
    if word.endswith(useful_suffixes):
        score += 3

    # Penalize obvious inflections unless the base still scores well.
    if word.endswith(("ed", "ing", "s")):
        score -= 1

    return score


def _tokenize_words(text: str) -> List[str]:
    """Tokenize story/article text into normalized word tokens."""
    return [
        normalized
        for raw in WORD_RE.findall(text or "")
        if (normalized := _normalize_word(raw))
    ]


def _sample_known_words(known_words: set, limit: int = 60) -> List[str]:
    """
    Take a deterministic spread of known words instead of the first alphabetic 60.

    This avoids only sampling words from the beginning of the alphabet.
    """
    cleaned = sorted({_normalize_word(str(w)) for w in known_words if str(w).strip()})
    if len(cleaned) <= limit:
        return cleaned

    step = len(cleaned) / limit
    return [cleaned[int(i * step)] for i in range(limit)]


def _safe_float_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class RAGOutput:
    title: str
    summary: str
    new_vocab: List[Dict[str, str]]
    difficulty_rating: float
    rationale: str
    coverage_ratio: float
    source_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class StoryOutput:
    title: str
    story: str
    new_vocab: List[Dict[str, str]]
    challenge_note: str
    vocab_level: str
    genre: str
    challenge: str
    known_word_ratio: float = 0.0
    unknown_word_ratio: float = 0.0
    new_word_count: int = 0
    total_word_count: int = 0
    target_known_range: Tuple[float, float] = (0.0, 1.0)
    within_target_range: bool = False
    actual_new_words: List[str] = field(default_factory=list)
    was_revised: bool = False


# ── Generator protocol ────────────────────────────────────────────────────────

class Generator(Protocol):
    """All generators must satisfy this interface."""

    def generate(
        self,
        query: Optional[str],
        context_chunks: List[ArticleChunk],
        new_words: List[str],
        vocab_level: str,
    ) -> RAGOutput: ...


# ── llama.cpp GPU generator ───────────────────────────────────────────────────

class LlamaCppGenerator:
    """
    Runs inference via llama-cpp-python with full GPU offloading.
    """

    def __init__(
        self,
        repo_id: str = "",
        filename: str = "",
        model_path: Optional[str] = None,
        n_gpu_layers: int = -1,
        tensor_split: Optional[List[float]] = None,
        context_length: int = 4096,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        from llama_cpp import Llama

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

        split: Optional[List[float]] = None
        if tensor_split is not None:
            if not tensor_split:
                raise ValueError("tensor_split must contain at least one value")
            if any(v <= 0 for v in tensor_split):
                raise ValueError("tensor_split values must be > 0")

            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible:
                visible_count = len([d for d in visible.split(",") if d.strip()])
                if visible_count > 0 and len(tensor_split) != visible_count:
                    raise ValueError(
                        "tensor_split length must match visible GPU count "
                        f"({len(tensor_split)} vs {visible_count})"
                    )

            total = float(sum(tensor_split))
            split = [float(v) / total for v in tensor_split]

        if model_path is not None:
            resolved_path = Path(model_path)
        else:
            resolved_path = Path(filename)
            if not resolved_path.is_absolute():
                resolved_path = Path.cwd() / resolved_path

        common_kwargs = dict(
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            low_vram=False,
            tensor_split=split,
            verbose=False,
        )

        if resolved_path.exists():
            self.llm = Llama(model_path=str(resolved_path), **common_kwargs)
        else:
            if not repo_id or not filename:
                raise ValueError(
                    f"Model file not found at '{resolved_path}' and no repo_id/filename "
                    "provided for download."
                )
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                **common_kwargs,
            )

        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(
        self,
        query: Optional[str],
        context_chunks: List[ArticleChunk],
        new_words: List[str],
        vocab_level: str,
    ) -> str:
        passages = "\n\n".join(
            f"[Article: {c.title}]\n{c.text[:600]}" for c in context_chunks[:3]
        )
        new_word_list = ", ".join(new_words[:15]) if new_words else "none"

        if query:
            interest_line = f"Student query / interest: {query}"
        else:
            interest_line = (
                "No specific topic requested — recommend this article based on "
                "how well its vocabulary matches the student's reading level."
            )

        return f"""You are a vocabulary-aware reading recommender for language learners.
Given a student's reading level and retrieved article passages, produce a structured JSON recommendation.

Student vocabulary level: {vocab_level}
{interest_line}
New words this student would encounter: {new_word_list}

Article passages:
{passages}

Respond ONLY with valid JSON in exactly this format (no extra text before or after):
{{
  "summary": "2-3 sentence summary of the article",
  "new_vocab": [
    {{"word": "example", "definition": "brief contextual definition"}}
  ],
  "difficulty_rating": 2.5,
  "rationale": "why this article is a good next read for this student"
}}
Use the same keys and value types, but replace the example values with content grounded in the provided passages.
Do not repeat literal placeholder phrases like "2-3 sentence summary of the article" or "example" unless they are truly in the source text."""

    @staticmethod
    def _extract_first_json_object(raw_text: str) -> dict:
        """Parse the first valid JSON object found in model output text."""
        decoder = json.JSONDecoder()

        start = raw_text.find("{")
        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(raw_text[start:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            start = raw_text.find("{", start + 1)

        raise ValueError("No valid JSON object found in model response")

    def generate(
        self,
        query: Optional[str],
        context_chunks: List[ArticleChunk],
        new_words: List[str],
        vocab_level: str,
    ) -> RAGOutput:
        """
        Generate a structured recommendation explanation for retrieved article chunks.
        This method is used by RAGPipeline.recommend().
        """
        top = context_chunks[0]
        prompt = self._build_prompt(query, context_chunks, new_words, vocab_level)

        try:
            response = self.llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=None,
            )
            raw_text = response["choices"][0]["text"].strip()
            parsed = self._extract_first_json_object(raw_text)

            summary = str(parsed.get("summary") or "").strip()
            rationale = str(parsed.get("rationale") or "").strip()

            parsed_vocab = parsed.get("new_vocab", [])
            if not isinstance(parsed_vocab, list):
                parsed_vocab = []

            try:
                difficulty_rating = float(parsed.get("difficulty_rating", 3.0))
            except (TypeError, ValueError):
                difficulty_rating = 3.0

            difficulty_rating = min(max(difficulty_rating, 1.0), 5.0)

            coverage = top.coverage_ratio.get(vocab_level, 0.0)
            try:
                coverage = float(coverage)
            except (TypeError, ValueError):
                coverage = 0.0

            if not rationale:
                rationale = (
                    f"Recommended for a {vocab_level.capitalize()} reader because "
                    f"its known-word coverage is {coverage:.1%}, making it a reasonable vocabulary match."
                )

            return RAGOutput(
                title=top.title,
                summary=summary,
                new_vocab=parsed_vocab,
                difficulty_rating=difficulty_rating,
                rationale=rationale,
                coverage_ratio=coverage,
                source_chunk_ids=[c.chunk_id for c in context_chunks],
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"[LlamaCppGenerator] Model response parse error: {e}") from e

    def _analyze_story_vocabulary(
        self,
        story: str,
        known_words: set,
        *,
        challenge: str,
        max_new_vocab: int,
    ) -> Dict[str, Any]:
        """
        Analyze generated story vocabulary against the student's known words.

        Coverage math uses all unknown words.
        actual_new_words is ranked and filtered to prioritize useful challenge words.
        """
        known_set = {
            _normalize_word(str(word))
            for word in known_words
            if str(word).strip()
        }

        tokens = _tokenize_words(story)
        total_word_count = len(tokens)

        known_count = sum(1 for token in tokens if _is_known_word(token, known_set))
        unknown_tokens = [token for token in tokens if not _is_known_word(token, known_set)]

        unique_unknown_words = sorted(set(unknown_tokens))
        ranked_unknown_words = sorted(
            unique_unknown_words,
            key=lambda word: (_challenge_word_score(word), len(word), word),
            reverse=True,
        )
        useful_challenge_words = [
            word for word in ranked_unknown_words if _challenge_word_score(word) >= 0
        ]

        known_word_ratio = _safe_float_ratio(known_count, total_word_count)
        unknown_word_ratio = 1.0 - known_word_ratio if total_word_count else 0.0

        target_known_range = STORY_TARGET_KNOWN_RANGES.get(
            challenge,
            STORY_TARGET_KNOWN_RANGES["medium"],
        )

        return {
            "total_word_count": total_word_count,
            "known_word_count": known_count,
            "unknown_token_count": len(unknown_tokens),
            "known_word_ratio": known_word_ratio,
            "unknown_word_ratio": unknown_word_ratio,
            "new_word_count": len(unique_unknown_words),
            "actual_new_words": useful_challenge_words[:max_new_vocab],
            "target_known_range": target_known_range,
            "within_target_range": (
                (target_known_range[0] - STORY_TARGET_TOLERANCE)
                <= known_word_ratio
                <= (target_known_range[1] + STORY_TARGET_TOLERANCE)
            ),
        }

    def _clean_story_new_vocab(
        self,
        model_new_vocab: Any,
        actual_new_words: List[str],
        *,
        max_new_vocab: int,
    ) -> List[Dict[str, str]]:
        """
        Keep only new_vocab entries that are actually useful unknown words in the story.

        If the model gives no useful vocab list, fall back to the ranked useful
        challenge words from the story.
        """
        actual_set = {_normalize_word(word) for word in actual_new_words}
        cleaned: List[Dict[str, str]] = []
        seen: set[str] = set()

        if isinstance(model_new_vocab, list):
            for item in model_new_vocab:
                if not isinstance(item, dict):
                    continue

                word = _normalize_word(str(item.get("word", "")))
                if not word or word in seen:
                    continue

                if _challenge_word_score(word) < 0:
                    continue

                # Keep only words that actually appear as useful unknown words in the story.
                if actual_set and word not in actual_set:
                    continue

                definition = str(item.get("definition", "")).strip()
                if not definition:
                    definition = "New word from the story; use context clues to infer its meaning."

                cleaned.append({
                    "word": word,
                    "definition": definition,
                })
                seen.add(word)

                if len(cleaned) >= max_new_vocab:
                    return cleaned

        # Fallback: include useful challenge words from the story.
        for word in actual_new_words:
            normalized = _normalize_word(word)
            if not normalized or normalized in seen:
                continue

            if _challenge_word_score(normalized) < 0:
                continue

            cleaned.append({
                "word": normalized,
                "definition": "New word from the story; use context clues to infer its meaning.",
            })
            seen.add(normalized)

            if len(cleaned) >= max_new_vocab:
                break

        return cleaned

    def _build_story_revision_prompt(
        self,
        *,
        title: str,
        story: str,
        model_new_vocab: List[Dict[str, str]],
        challenge_note: str,
        vocab_level: str,
        known_words: set,
        topic: Optional[str],
        genre: str,
        challenge: str,
        target_words: int,
        vocab_stats: Dict[str, Any],
    ) -> str:
        """
        Build a prompt that asks the model to revise a story toward the target
        known-word coverage range.
        """
        target_low, target_high = vocab_stats["target_known_range"]
        known_ratio = vocab_stats["known_word_ratio"]
        actual_new_words = ", ".join(vocab_stats.get("actual_new_words", [])[:20])

        sample_known = _sample_known_words(known_words, limit=80)
        known_sample_str = ", ".join(sample_known) if sample_known else "(none provided)"

        if known_ratio < target_low:
            direction = (
                "The story is too difficult for this student. Rewrite it to be easier. "
                "Use mostly common, familiar words from the known vocabulary sample. "
                "Use short, clear sentences. Replace advanced or rare words with simpler words. "
                "Keep only 5 to 8 challenging words total. "
                "Avoid using many science, abstract, or poetic words. "
                "The revised story must be closer to the target known-word coverage range."
            )
        elif known_ratio > target_high:
            direction = (
                "The story is too easy for this student. Revise it to add a few slightly "
                "harder and meaningful words while keeping the story understandable. "
                "Prefer useful challenge words over basic words."
            )
        else:
            direction = (
                "The story is already close to the target difficulty. Make only light edits."
            )

        return f"""Return ONLY valid JSON. Do not use markdown. Do not explain.

You are revising a story for a language learner.

Student reading level: {vocab_level}
Genre: {genre}
Topic: {topic or "an interesting adventure"}
Challenge setting: {challenge}
Target length: about {target_words} words

Target known-word coverage range: {target_low:.0%} to {target_high:.0%}
Current known-word coverage: {known_ratio:.1%}
Current useful challenge words include: {actual_new_words or "none"}

Revision instruction:
{direction}

Important: do not preserve difficult wording from the original if simpler wording works better.
Important: if you include challenge words, choose meaningful words worth learning, not basic words like happy, lake, clapped, danced, or met.

Use many familiar words from this student's known vocabulary sample:
{known_sample_str}

Original title:
{title}

Original story:
{story}

Original vocabulary note:
{challenge_note}

Return JSON using exactly these keys:
{{
  "title": "Revised story title",
  "story": "Complete revised story text of about {target_words} words",
  "new_vocab": [
    {{"word": "new word", "definition": "short definition"}}
  ],
  "challenge_note": "Short note explaining how the revision fits the target difficulty"
}}"""

    def generate_story(
        self,
        vocab_level: str,
        known_words: set,
        *,
        topic: Optional[str] = None,
        genre: str = "adventure",
        challenge: str = "medium",
        target_words: int = 400,
        max_new_vocab: int = 10,
    ) -> StoryOutput:
        """
        Generate a short story calibrated to the student's vocabulary level.

        It retries once if the model returns valid JSON but no usable story text.
        It also revises once if the generated story misses the target coverage range.
        """
        challenge = (challenge or "medium").strip().lower()
        if challenge not in STORY_TARGET_KNOWN_RANGES:
            challenge = "medium"

        _CHALLENGE_PROMPTS = {
            "low": (
                "Use almost exclusively words the student already knows. "
                "Introduce at most 2-3 useful challenge words, and define each one clearly in context."
            ),
            "medium": (
                "Use mostly familiar vocabulary but weave in 5-8 useful, slightly harder words "
                "that are natural for this level. Choose meaningful words worth learning, not basic words."
            ),
            "high": (
                f"Stretch the student's vocabulary by including up to {max_new_vocab} useful challenge words "
                "that are one level above their current level. Make meaning guessable from context."
            ),
        }

        sample_known = _sample_known_words(known_words, limit=60)
        known_sample_str = ", ".join(sample_known) if sample_known else "(none provided)"

        topic_line = (
            f"Topic / theme: {topic}"
            if topic
            else "Topic: choose something interesting for this student."
        )
        challenge_instruction = _CHALLENGE_PROMPTS[challenge]

        base_prompt = f"""You are a creative writing tutor crafting a short story for a language learner.

Student reading level: {vocab_level}
Genre: {genre}
{topic_line}
Target story length: approximately {target_words} words

Vocabulary challenge setting: {challenge}
{challenge_instruction}

A sample of words this student already knows. Use many of these familiar words naturally:
{known_sample_str}

Write a complete story. The story should be slightly above the student's current reading level, but still understandable. Most words should be familiar. Do not make the story sound advanced just because the topic is science fiction.
If you include challenge words, choose useful words worth learning. Avoid listing basic words like happy, lake, clapped, danced, or met as new vocabulary.

Respond ONLY with valid JSON in exactly this format. No markdown. No explanation before or after:
{{
  "title": "Story title",
  "story": "Full story text here, about {target_words} words",
  "new_vocab": [
    {{"word": "example", "definition": "brief definition as used in the story"}}
  ],
  "challenge_note": "One sentence describing how this story challenges the student"
}}"""

        retry_prompt = f"""Return ONLY valid JSON. Do not use markdown. Do not explain.

Write a complete {genre} story for a {vocab_level} reader.

Topic: {topic or "an interesting adventure"}
Length: about {target_words} words.
Challenge: {challenge}

The story must be slightly above the student's current reading level, but still understandable.
Include a small number of useful challenge words. Do not list basic words like happy, lake, clapped, danced, or met as new vocabulary.

Use exactly these JSON keys:
{{
  "title": "Story title",
  "story": "Complete story text of about {target_words} words",
  "new_vocab": [
    {{"word": "new word", "definition": "short definition"}}
  ],
  "challenge_note": "Short note about why this story is a good challenge"
}}"""

        last_error: Optional[Exception] = None

        for attempt_number, prompt in enumerate([base_prompt, retry_prompt], start=1):
            try:
                response = self.llm(
                    prompt,
                    max_tokens=max(self.max_tokens, target_words * 3),
                    temperature=max(self.temperature, 0.7),
                    stop=None,
                )
                raw_text = response["choices"][0]["text"].strip()
                parsed = self._extract_first_json_object(raw_text)

                title = str(parsed.get("title") or "").strip()
                story = str(
                    parsed.get("story")
                    or parsed.get("body")
                    or parsed.get("text")
                    or parsed.get("content")
                    or ""
                ).strip()

                new_vocab = parsed.get("new_vocab") or parsed.get("vocabulary") or []
                if not isinstance(new_vocab, list):
                    new_vocab = []

                challenge_note = str(
                    parsed.get("challenge_note")
                    or parsed.get("note")
                    or ""
                ).strip()

                if not title:
                    title = "Untitled"

                if len(story.split()) >= 50:
                    vocab_stats = self._analyze_story_vocabulary(
                        story,
                        known_words,
                        challenge=challenge,
                        max_new_vocab=max_new_vocab,
                    )

                    revised = False

                    if not vocab_stats["within_target_range"]:
                        revision_prompt = self._build_story_revision_prompt(
                            title=title,
                            story=story,
                            model_new_vocab=new_vocab,
                            challenge_note=challenge_note,
                            vocab_level=vocab_level,
                            known_words=known_words,
                            topic=topic,
                            genre=genre,
                            challenge=challenge,
                            target_words=target_words,
                            vocab_stats=vocab_stats,
                        )

                        revision_response = self.llm(
                            revision_prompt,
                            max_tokens=max(self.max_tokens, target_words * 3),
                            temperature=max(self.temperature, 0.6),
                            stop=None,
                        )
                        revision_raw = revision_response["choices"][0]["text"].strip()
                        revision_parsed = self._extract_first_json_object(revision_raw)

                        revised_title = str(revision_parsed.get("title") or title).strip()
                        revised_story = str(
                            revision_parsed.get("story")
                            or revision_parsed.get("body")
                            or revision_parsed.get("text")
                            or revision_parsed.get("content")
                            or ""
                        ).strip()
                        revised_new_vocab = (
                            revision_parsed.get("new_vocab")
                            or revision_parsed.get("vocabulary")
                            or []
                        )
                        revised_challenge_note = str(
                            revision_parsed.get("challenge_note")
                            or revision_parsed.get("note")
                            or challenge_note
                            or ""
                        ).strip()

                        # Accept the revision only if it is a real story.
                        if len(revised_story.split()) >= 50:
                            title = revised_title or title
                            story = revised_story
                            new_vocab = revised_new_vocab if isinstance(revised_new_vocab, list) else []
                            challenge_note = revised_challenge_note
                            revised = True
                            # If the accepted revision is still too far outside the target,
                            # try one stricter repair pass.
                            if not vocab_stats["within_target_range"]:
                                repair_prompt = f"""Return ONLY valid JSON. Do not use markdown. Do not explain.

Rewrite this story so it better fits a {vocab_level} reader.

Rules:
- Keep the same topic and basic plot.
- Use short, clear sentences.
- Use common familiar words whenever possible.
- Keep only 5 to 8 useful challenge words.
- Do not use many advanced science, abstract, or poetic words.
- Target length: about {target_words} words.
- Make the story closer to the target known-word coverage range.

Original story:
{story}

Return JSON exactly like this:
{{
  "title": "Story title",
  "story": "Complete rewritten story of about {target_words} words",
  "new_vocab": [
    {{"word": "useful challenge word", "definition": "short definition"}}
  ],
  "challenge_note": "Why this version fits the student's level"
}}"""

                                repair_response = self.llm(
                                    repair_prompt,
                                    max_tokens=max(self.max_tokens, target_words * 3),
                                    temperature=0.4,
                                    stop=None,
                                )
                                repair_raw = repair_response["choices"][0]["text"].strip()
                                repair_parsed = self._extract_first_json_object(repair_raw)

                                repair_title = str(repair_parsed.get("title") or title).strip()
                                repair_story = str(
                                    repair_parsed.get("story")
                                    or repair_parsed.get("body")
                                    or repair_parsed.get("text")
                                    or repair_parsed.get("content")
                                    or ""
                                ).strip()
                                repair_new_vocab = (
                                    repair_parsed.get("new_vocab")
                                    or repair_parsed.get("vocabulary")
                                    or []
                                )
                                repair_challenge_note = str(
                                    repair_parsed.get("challenge_note")
                                    or repair_parsed.get("note")
                                    or challenge_note
                                    or ""
                                ).strip()

                                if len(repair_story.split()) >= 50:
                                    repair_stats = self._analyze_story_vocabulary(
                                        repair_story,
                                        known_words,
                                        challenge=challenge,
                                        max_new_vocab=max_new_vocab,
                                    )

                                    # Accept repair if it improves the known-word ratio
                                    # toward the target lower bound.
                                    current_distance = abs(
                                        vocab_stats["known_word_ratio"]
                                        - vocab_stats["target_known_range"][0]
                                    )
                                    repair_distance = abs(
                                        repair_stats["known_word_ratio"]
                                        - repair_stats["target_known_range"][0]
                                    )

                                    if repair_stats["within_target_range"] or repair_distance < current_distance:
                                        title = repair_title or title
                                        story = repair_story
                                        new_vocab = repair_new_vocab if isinstance(repair_new_vocab, list) else []
                                        challenge_note = repair_challenge_note
                                        vocab_stats = repair_stats
                            vocab_stats = self._analyze_story_vocabulary(
                                story,
                                known_words,
                                challenge=challenge,
                                max_new_vocab=max_new_vocab,
                            )
                        else:
                            # If the model failed to revise properly, try one stricter repair prompt.
                            repair_prompt = f"""Return ONLY valid JSON. Do not use markdown. Do not explain.

                        Rewrite this story so it is easier for a {vocab_level} reader.

                        Rules:
                        - Keep the same topic and plot.
                        - Use short, clear sentences.
                        - Use common words whenever possible.
                        - Keep only 5 to 8 useful challenge words.
                        - Do not use many advanced science words.
                        - Target length: about {target_words} words.

                        Original story:
                        {story}

                        Return JSON exactly like this:
                        {{
                          "title": "Story title",
                          "story": "Complete rewritten story of about {target_words} words",
                          "new_vocab": [
                            {{"word": "useful challenge word", "definition": "short definition"}}
                          ],
                          "challenge_note": "Why this version fits the student's level"
                        }}"""

                            repair_response = self.llm(
                                repair_prompt,
                                max_tokens=max(self.max_tokens, target_words * 3),
                                temperature=0.4,
                                stop=None,
                            )
                            repair_raw = repair_response["choices"][0]["text"].strip()
                            repair_parsed = self._extract_first_json_object(repair_raw)

                            repair_title = str(repair_parsed.get("title") or title).strip()
                            repair_story = str(
                                repair_parsed.get("story")
                                or repair_parsed.get("body")
                                or repair_parsed.get("text")
                                or repair_parsed.get("content")
                                or ""
                            ).strip()
                            repair_new_vocab = repair_parsed.get("new_vocab") or repair_parsed.get("vocabulary") or []
                            repair_challenge_note = str(
                                repair_parsed.get("challenge_note")
                                or repair_parsed.get("note")
                                or challenge_note
                                or ""
                            ).strip()

                            if len(repair_story.split()) >= 50:
                                title = repair_title or title
                                story = repair_story
                                new_vocab = repair_new_vocab if isinstance(repair_new_vocab, list) else []
                                challenge_note = repair_challenge_note
                                revised = True

                                vocab_stats = self._analyze_story_vocabulary(
                                    story,
                                    known_words,
                                    challenge=challenge,
                                    max_new_vocab=max_new_vocab,
                                )
                    cleaned_new_vocab = self._clean_story_new_vocab(
                        new_vocab,
                        vocab_stats["actual_new_words"],
                        max_new_vocab=max_new_vocab,
                    )

                    return StoryOutput(
                        title=title,
                        story=story,
                        new_vocab=cleaned_new_vocab,
                        challenge_note=challenge_note,
                        vocab_level=vocab_level,
                        genre=genre,
                        challenge=challenge,
                        known_word_ratio=vocab_stats["known_word_ratio"],
                        unknown_word_ratio=vocab_stats["unknown_word_ratio"],
                        new_word_count=vocab_stats["new_word_count"],
                        total_word_count=vocab_stats["total_word_count"],
                        target_known_range=vocab_stats["target_known_range"],
                        within_target_range=vocab_stats["within_target_range"],
                        actual_new_words=vocab_stats["actual_new_words"],
                        was_revised=revised,
                    )

                last_error = RuntimeError(
                    f"Attempt {attempt_number} returned too little story text "
                    f"({len(story.split())} words)."
                )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = e

        raise RuntimeError(
            "[LlamaCppGenerator] Story generation failed after retry. "
            f"Last error: {last_error}"
        )


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(
        self,
        retriever: TwoStageRetriever,
        generator: Generator,
        coverage_window: Tuple[float, float] = (0.85, 0.97),
    ):
        self.retriever = retriever
        self.generator = generator
        self.coverage_window = coverage_window

    def recommend(
        self,
        query: Optional[str],
        vocab_level: str,
        top_k: int = 5,
        student_profile: Optional["StudentProfile"] = None,
        mark_as_read: bool = False,
    ) -> List[RAGOutput]:
        ranked = self.retriever.retrieve(
            query=query,
            vocab_level=vocab_level,
            coverage_window=self.coverage_window,
            student_profile=student_profile,
        )

        seen: Dict[str, Tuple[ArticleChunk, float]] = {}
        for chunk, score in ranked:
            if chunk.article_id not in seen or score > seen[chunk.article_id][1]:
                seen[chunk.article_id] = (chunk, score)

        outputs: List[RAGOutput] = []
        for chunk, _ in list(seen.values())[:top_k]:
            if student_profile is not None:
                new_words = student_profile.get_remaining_new_words(chunk)
            else:
                new_words = chunk.new_words.get(vocab_level, [])

            output = self.generator.generate(
                query=query,
                context_chunks=[chunk],
                new_words=new_words,
                vocab_level=vocab_level,
            )
            # Use the same student-specific coverage that the reranker used.
            # This prevents the API from saving/displaying stale static article coverage.
            if student_profile is not None:
                try:
                    output.coverage_ratio = float(student_profile.adjusted_coverage(chunk))
                except (TypeError, ValueError):
                    pass

            outputs.append(output)

            if mark_as_read and student_profile is not None:
                student_profile.mark_as_read(chunk)

        return outputs