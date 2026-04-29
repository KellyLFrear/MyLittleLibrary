from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Tuple
from pathlib import Path

from src.embeddings.chunker import ArticleChunk
from src.rag.retriever import TwoStageRetriever

if TYPE_CHECKING:
    from src.rag.student_profile import StudentProfile


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class RAGOutput:
    title: str
    summary: str
    new_vocab: List[Dict[str, str]]  # [{"word": "...", "definition": "..."}]
    difficulty_rating: float         # 1.0–5.0
    rationale: str
    coverage_ratio: float
    source_chunk_ids: List[str] = field(default_factory=list)


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

    Install (CUDA 12.x):
        pip install llama-cpp-python \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

    Usage:
        generator = LlamaCppGenerator(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        )

    Parameters
    ----------
    repo_id : str
        Hugging Face repository ID for the GGUF model.
    filename : str
        GGUF filename within the repository.
    n_gpu_layers : int
        Number of transformer layers to offload to GPU. -1 offloads all layers.
    context_length : int
        Maximum context window in tokens (must match the model's capability).
    temperature : float
        Sampling temperature for generation (lower = more deterministic).
    max_tokens : int
        Maximum number of tokens to generate per call.
    """

    def __init__(
        self,
        repo_id: str,
        filename: str,
        n_gpu_layers: int = -1,
        tensor_split: Optional[List[float]] = None,
        context_length: int = 4096,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        from llama_cpp import Llama

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

        local_path = Path(filename)
        if not local_path.is_absolute():
            local_path = Path.cwd() / local_path

        common_kwargs = dict(
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            low_vram=False,
            tensor_split=split,
            verbose=False,
        )
        if local_path.exists():
            self.llm = Llama(model_path=str(local_path), **common_kwargs)
        else:
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

            return RAGOutput(
                title=top.title,
                summary=parsed.get("summary", ""),
                new_vocab=parsed.get("new_vocab", []),
                difficulty_rating=float(parsed.get("difficulty_rating", 3.0)),
                rationale=parsed.get("rationale", ""),
                coverage_ratio=top.coverage_ratio.get(vocab_level, 0.0),
                source_chunk_ids=[c.chunk_id for c in context_chunks],
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"[LlamaCppGenerator] Model response parse error: {e}") from e

    # ── Story generation ──────────────────────────────────────────────────────

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
    ) -> "StoryOutput":
        """
        Generate a short story calibrated to the student's vocabulary level.

        Parameters
        ----------
        vocab_level : str
            ``"beginner"``, ``"intermediate"``, or ``"advanced"``.
        known_words : set
            The student's known word set (from ``StudentProfile``).
        topic : str, optional
            Optional topic or theme hint (e.g. ``"the ocean"``).
        genre : str
            Story genre: ``"adventure"``, ``"mystery"``, ``"fantasy"``,
            ``"sci-fi"``, or ``"slice-of-life"``.
        challenge : str
            How hard to push the student:
            - ``"low"``    — almost all known words, 1-3 new words
            - ``"medium"`` — mostly familiar, ~5-8 new words  (default)
            - ``"high"``   — noticeably challenging, up to ``max_new_vocab`` new words
        target_words : int
            Approximate target word count for the story body (default 400).
        max_new_vocab : int
            Hard cap on the number of new words to introduce (default 10).
        """
        _CHALLENGE_PROMPTS = {
            "low": (
                "Use almost exclusively words the student already knows. "
                "Introduce at most 2-3 new words, and define each one clearly in context."
            ),
            "medium": (
                "Use mostly familiar vocabulary but weave in 5-8 new, slightly harder words "
                "that are natural for this level. Clarify meaning through context, not footnotes."
            ),
            "high": (
                f"Stretch the student's vocabulary by including up to {max_new_vocab} new words "
                "that are one level above their current level. Make meaning guessable from context."
            ),
        }

        # Sample a representative subset of known words to anchor the model
        sample_known = sorted(known_words)[:60]
        known_sample_str = ", ".join(sample_known) if sample_known else "(none provided)"

        topic_line = f"Topic / theme: {topic}" if topic else "Topic: choose something interesting for this student."
        challenge_instruction = _CHALLENGE_PROMPTS.get(challenge, _CHALLENGE_PROMPTS["medium"])

        prompt = f"""You are a creative writing tutor crafting a short story for a language learner.

Student reading level: {vocab_level}
Genre: {genre}
{topic_line}
Target story length: approximately {target_words} words

Vocabulary challenge setting: {challenge}
{challenge_instruction}

A sample of words this student already knows (use freely):
{known_sample_str}

Write the story first, then at the end provide a brief vocabulary note for any new or challenging words you used.

Respond ONLY with valid JSON in exactly this format (no extra text before or after):
{{
  "title": "Story title",
  "story": "Full story text here (~{target_words} words)",
  "new_vocab": [
    {{"word": "example", "definition": "brief definition as used in the story"}}
  ],
  "challenge_note": "One sentence describing how this story challenges the student"
}}"""

        try:
            response = self.llm(
                prompt,
                max_tokens=max(self.max_tokens, target_words * 2),
                temperature=max(self.temperature, 0.7),  # more creative for stories
                stop=None,
            )
            raw_text = response["choices"][0]["text"].strip()
            parsed = self._extract_first_json_object(raw_text)

            return StoryOutput(
                title=parsed.get("title", "Untitled"),
                story=parsed.get("story", ""),
                new_vocab=parsed.get("new_vocab", []),
                challenge_note=parsed.get("challenge_note", ""),
                vocab_level=vocab_level,
                genre=genre,
                challenge=challenge,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"[LlamaCppGenerator] Story response parse error: {e}") from e


@dataclass
class StoryOutput:
    title: str
    story: str
    new_vocab: List[Dict[str, str]]
    challenge_note: str
    vocab_level: str
    genre: str
    challenge: str


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

        # De-duplicate: keep the highest-scoring chunk per article
        seen: Dict[str, Tuple[ArticleChunk, float]] = {}
        for chunk, score in ranked:
            if chunk.article_id not in seen or score > seen[chunk.article_id][1]:
                seen[chunk.article_id] = (chunk, score)

        outputs: List[RAGOutput] = []
        for chunk, _ in list(seen.values())[:top_k]:
            # Use profile-filtered new words if profile is available
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
            outputs.append(output)

            # Simulate vocab growth: add new words to student's known list
            if mark_as_read and student_profile is not None:
                student_profile.mark_as_read(chunk)

        return outputs
