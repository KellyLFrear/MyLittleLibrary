from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Tuple

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
        query: str,
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
        generator = LlamaCppGenerator(model_path="models/llama-3.1-8b-instruct.Q4_K_M.gguf")

    Parameters
    ----------
    model_path : str
        Path to a GGUF model file (e.g. llama-3.1-8b-instruct.Q4_K_M.gguf).
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
        model_path: str,
        n_gpu_layers: int = -1,
        context_length: int = 4096,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=context_length,
            verbose=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(
        self,
        query: str,
        context_chunks: List[ArticleChunk],
        new_words: List[str],
        vocab_level: str,
    ) -> str:
        passages = "\n\n".join(
            f"[Article: {c.title}]\n{c.text[:600]}" for c in context_chunks[:3]
        )
        new_word_list = ", ".join(new_words[:15]) if new_words else "none"

        return f"""You are a vocabulary-aware reading recommender for language learners.
Given a student's query and retrieved article passages, produce a structured JSON recommendation.

Student vocabulary level: {vocab_level}
Student query / interest: {query}
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
}}"""

    def generate(
        self,
        query: str,
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

            # Extract JSON block even if the model adds preamble text
            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in model response")
            parsed = json.loads(raw_text[start:end])

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
            print(f"[LlamaCppGenerator] Warning: fell back to template ({e})")
            return TemplateGenerator().generate(query, context_chunks, new_words, vocab_level)


# ── Template generator (no model required) ───────────────────────────────────

class TemplateGenerator:
    """
    Rule-based fallback — use this to test the pipeline before any model is set up.
    Produces valid, parseable output so retrieval and reranking can be verified
    independently of the generative component.
    """

    def generate(
        self,
        query: str,
        context_chunks: List[ArticleChunk],
        new_words: List[str],
        vocab_level: str,
    ) -> RAGOutput:
        top = context_chunks[0]
        summary = " ".join(top.text.split()[:80]) + "..."
        new_vocab = [
            {"word": w, "definition": "(definition pending model)"}
            for w in new_words[:10]
        ]
        ratio = top.coverage_ratio.get(vocab_level, 0.0)
        difficulty = round(1 + 4 * (1 - ratio), 1)
        return RAGOutput(
            title=top.title,
            summary=summary,
            new_vocab=new_vocab,
            difficulty_rating=max(1.0, min(5.0, difficulty)),
            rationale=(
                f"Matches your interest in '{query}'. "
                f"At {ratio:.0%} coverage it introduces {len(new_words)} learnable new words."
            ),
            coverage_ratio=ratio,
            source_chunk_ids=[c.chunk_id for c in context_chunks],
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
        query: str,
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
