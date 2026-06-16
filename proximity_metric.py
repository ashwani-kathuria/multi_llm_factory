"""
proximity_metric.py
===================
Topic-Aware Self-Correction Proximity algorithm for LLM uncertainty quantification.

Detects hedge expressions in a reasoning trace, finds subsequent verification
statements, measures semantic subject similarity via SentenceTransformers, and
adjusts per-hedge uncertainty weights based on whether each hedge was resolved.

Public interface
----------------
    from proximity_metric import calculate_proximity_metrics

    result = calculate_proximity_metrics(reasoning_text)

The returned dict contains:
    {
        "total_hedges"    : int,
        "resolved_hedges" : int,
        "unresolved_hedges": int,
        "trur"            : float,   # resolved / total
        "weighted_trur"   : float,   # mean weight reduction fraction
        "matches"         : [...]    # per-hedge detail records
    }

Configuration
-------------
Set PROXIMITY_EMBEDDING_MODEL in your .env (or pass embedding_model= directly)
to select any SentenceTransformer model, e.g.:
    all-MiniLM-L6-v2  (default — fast, CPU-friendly)
    all-mpnet-base-v2
    multi-qa-MiniLM-L6-cos-v1
    bge-small-en-v1.5  /  bge-base-en-v1.5
    e5-small-v2        /  e5-base-v2
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (loaded at most once per process)
# ---------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, object] = {}   # embedding_model_name -> SentenceTransformer
_NLP_CACHE:   Dict[str, object] = {}   # lang_model_name      -> spacy.Language
_EMBEDDING_CACHE: Dict[Tuple[str, str], np.ndarray] = {}  # (model_name, text) -> vector

# ---------------------------------------------------------------------------
# Keyword lists (Steps 1 & 2)
# ---------------------------------------------------------------------------
_HEDGE_KEYWORDS: List[str] = [
    # ── Core epistemic modals (original) ─────────────────────────────────────
    "maybe", "perhaps", "might", "could", "possibly", "probably",
    "i think", "i believe", "not sure", "uncertain", "seems", "appears",
    "likely", "unlikely",
    # ── Original additions ────────────────────────────────────────────────────
    "assume",
    "aim for",
    "decent",
    "typically",
    "let's assume",   # longer phrase before "assume"
    "let's say",
    # ── Stronger uncertainty (longer phrases first) ───────────────────────────
    "i'm not certain",        # before "not certain"
    "i can't be sure",
    "i cannot be sure",
    "not entirely sure",      # before "not sure" (already listed, no conflict)
    "not entirely clear",
    "not certain",
    "not obviously",
    "not obvious",
    "i'm unsure",
    "i am unsure",
    "hard to say",
    "difficult to determine",
    "unclear",
    "ambiguous",
    # ── Reconsideration markers (LLM chain-of-thought specific) ──────────────
    "on second thought",      # longer phrase first
    "let me reconsider",
    "i need to reconsider",
    "i might be wrong",
    "i may have made an error",
    "hold on,",               # comma anchors to avoid "hold on tight"
    "wait,",                  # comma anchors to avoid "wait time"
    "hmm,",
    # ── Possibility phrases (longer before shorter) ───────────────────────────
    "there's a possibility",
    "there is a possibility",
    "one possibility",
    "another possibility",
    "it's possible",
    "it is possible",
    "there may be",
    "alternatively",
    # ── Approximation markers ─────────────────────────────────────────────────
    "approximately",
    "roughly",
    "more or less",
    # ── Epistemic / belief markers ────────────────────────────────────────────
    "i suspect",
    "i suppose",
    "i wonder",
    "i'd guess",
    "in my estimation",
    "not necessarily",
    "to some extent",
    "considering",            # kept from original — dual-use word
]

_VERIFICATION_KEYWORDS: List[str] = [
    # ── Core verification verbs (original) ───────────────────────────────────
    "verify", "check", "confirm", "recalculate", "validate", "test",
    "review", "inspect", "determine", "conclude", "therefore", "indeed",
    "confirmed",
    # ── Original additions ────────────────────────────────────────────────────
    "make sure",
    "count the lines",
    "double-check",
    "accuracy",
    "refine",
    "perfect",
    "done",
    # ── Logical connectives (conclusion derivation) ───────────────────────────
    "it follows that",        # longest phrase first
    "as a result",
    "consequently",
    "this implies",
    "this indicates",
    "therefore",              # already listed; kept for completeness
    "hence",
    "thus",
    # ── Explicit conclusion markers (longer phrases first) ────────────────────
    "the conclusion is",
    "in conclusion",
    "to conclude",
    "to summarize",
    "in summary",
    "the answer is",
    "the result is",
    "we can conclude",
    "i can conclude",
    "i conclude",
    # ── Verification intent phrases (LLM action phrases) ──────────────────────
    "let me recalculate",     # longer before "let me check" etc.
    "let me recompute",
    "let me re-examine",
    "let me calculate",
    "let me compute",
    "let me confirm",
    "let me examine",
    "let me verify",
    "let me check",
    "i will verify",
    "i should verify",
    "i should check",
    # ── Consistency & evidence markers ────────────────────────────────────────
    "upon closer inspection",
    "upon examination",
    "upon reflection",
    "after reviewing",
    "after checking",
    "after analysis",
    "having confirmed",
    "having verified",
    "consistent with",
    "this is consistent",
    "aligns with",
    "corroborates",
    "as demonstrated",
    "as shown",
    # ── Confidence & certainty assertions ─────────────────────────────────────
    "i am confident",         # before "i'm confident" in case of contraction
    "i'm confident",
    "i can confirm",
    "it is confirmed",
    "it is evident",
    "it is clear",
    "without doubt",
    "no doubt",
    "considering",            # dual-use: kept from original for verification context
]

# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------
@dataclass
class _Statement:
    id: str
    sentence: str
    subject: str
    position: int   # sentence index (0-based)


@dataclass
class _HedgeResult:
    id: str
    subject: str
    position: int
    sentence: str = ""              # the original sentence that triggered detection
    matched_keyword: str = ""       # the keyword that fired the hedge detector
    matched_verification: Optional[str] = None
    subject_similarity: Optional[float] = None
    proximity_score: Optional[float] = None
    match_score: Optional[float] = None
    resolved: bool = False
    effective_weight: float = 1.0


# ---------------------------------------------------------------------------
# Singleton loaders
# ---------------------------------------------------------------------------
def _load_embedding_model(model_name: str):
    """
    Load a SentenceTransformer model, reusing the in-process cached instance.

    Strategy:
      1. Try ``local_files_only=True`` — loads instantly from the HuggingFace
         disk cache with zero network requests (warm path).
      2. On first use (cache miss / OSError) fall back to a normal download
         which populates the disk cache for all future calls (cold path).

    Verbose httpx / HuggingFace Hub INFO logs are suppressed during loading
    so they don't flood the application terminal.
    """
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        # Silence the per-request httpx / huggingface_hub chatter
        _hf_loggers = [
            logging.getLogger("httpx"),
            logging.getLogger("httpcore"),
            logging.getLogger("huggingface_hub"),
            logging.getLogger("huggingface_hub.file_download"),
            logging.getLogger("huggingface_hub.repocard"),
        ]
        _saved_levels = [lg.level for lg in _hf_loggers]
        for lg in _hf_loggers:
            lg.setLevel(logging.WARNING)

        try:
            # --- Warm path: model already on disk, no network needed ---
            logger.info(
                "[proximity_metric] Loading embedding model '%s' from local cache.",
                model_name,
            )
            model = SentenceTransformer(model_name, local_files_only=True)
            logger.info("[proximity_metric] Embedding model loaded from cache.")
        except Exception:
            # --- Cold path: first-time download ---
            logger.info(
                "[proximity_metric] Local cache miss — downloading '%s' from "
                "HuggingFace Hub (one-time only).",
                model_name,
            )
            model = SentenceTransformer(model_name, local_files_only=False)
            logger.info(
                "[proximity_metric] Embedding model downloaded and cached locally."
            )
        finally:
            # Restore original log levels
            for lg, lvl in zip(_hf_loggers, _saved_levels):
                lg.setLevel(lvl)

        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def _load_nlp():
    """Load spaCy's small English model once, reusing the cached instance."""
    lang = "en_core_web_sm"
    if lang not in _NLP_CACHE:
        try:
            import spacy  # type: ignore
            _NLP_CACHE[lang] = spacy.load(lang)
            logger.info("[proximity_metric] spaCy model '%s' loaded.", lang)
        except OSError:
            logger.warning(
                "[proximity_metric] spaCy model '%s' not found. "
                "Run: python -m spacy download %s. "
                "Falling back to regex subject extraction.",
                lang, lang,
            )
            _NLP_CACHE[lang] = None
        except ImportError:
            logger.warning(
                "[proximity_metric] spaCy not installed. "
                "Falling back to regex subject extraction."
            )
            _NLP_CACHE[lang] = None
    return _NLP_CACHE[lang]


# ---------------------------------------------------------------------------
# Step 1 — Detect hedge statements
# ---------------------------------------------------------------------------
def _detect_hedges(sentences: List[str]) -> List[_Statement]:
    """Return a _Statement for every sentence that contains a hedge keyword.

    The _Statement.subject field is left empty here (filled in Step 3).
    The first matching keyword is stored in a side-table returned alongside
    the statements so callers can record which keyword fired.

    Verification-dominance rule: if a sentence already contains ≥2 verification
    keywords it is acting as a verification statement, not a hedge, and is
    skipped here even if a hedge keyword also appears in it.
    """
    hedges: List[_Statement] = []
    hedge_index = 0
    for pos, sent in enumerate(sentences):
        lower = sent.lower()

        # Verification-dominance: skip sentences that are clearly verification
        ver_kw_count = sum(1 for vkw in _VERIFICATION_KEYWORDS if vkw in lower)
        if ver_kw_count >= 2:
            continue

        for kw in _HEDGE_KEYWORDS:
            if kw in lower:
                hedge_index += 1
                stmt = _Statement(
                    id=f"H{hedge_index}",
                    sentence=sent.strip(),
                    subject="",          # filled in Step 3
                    position=pos,
                )
                # Stash the triggering keyword as a custom attribute so
                # _assign_resolutions can carry it into _HedgeResult.
                stmt._matched_keyword = kw  # type: ignore[attr-defined]
                hedges.append(stmt)
                break   # only record a sentence once even if multiple keywords match
    return hedges


# ---------------------------------------------------------------------------
# Step 2 — Detect verification statements
# ---------------------------------------------------------------------------
def _detect_verifications(sentences: List[str]) -> List[_Statement]:
    """Return a _Statement for every sentence that contains a verification keyword."""
    verifications: List[_Statement] = []
    ver_index = 0
    for pos, sent in enumerate(sentences):
        lower = sent.lower()
        if any(kw in lower for kw in _VERIFICATION_KEYWORDS):
            ver_index += 1
            verifications.append(_Statement(
                id=f"V{ver_index}",
                sentence=sent.strip(),
                subject="",          # filled in Step 3
                position=pos,
            ))
    return verifications


# ---------------------------------------------------------------------------
# Step 3 — Subject extraction
# ---------------------------------------------------------------------------
def _extract_subject_spacy(sentence: str, nlp, matched_keyword: str = "") -> str:
    """
    Use spaCy dependency parsing to extract the primary subject noun phrase.

    Strategy (in priority order):
      0.  Prepositional uncertainty target: tokens matching hedge adjectives
          (sure, certain, confident) that have a 'prep' child → return the
          pobj of that prep.  Handles "not sure about X", "uncertain about X".
      1.  Object of a verb with a hedge/verification keyword as the head
          (e.g. "verify the equation" → "equation").
      1.5 Keyword-proximate noun chunk: locate the triggering keyword's token
          position and return the first non-pronoun noun chunk that starts
          within 3 tokens after it.  Handles "maybe a dash of symbolism" →
          "a dash of symbolism".  Falls back to a 2-token text slice when no
          noun chunk is found immediately after the keyword.
      2.  Non-pronoun nominal subject of the root verb.
      3.  First non-pronoun noun chunk in the sentence.
      4.  Fallback: regex heuristic.
    """
    doc = nlp(sentence)

    # Priority 0: adjective hedge with prepositional target
    # e.g. "I'm not sure about the boundary condition"
    #       sure(acomp) -[prep]-> about -[pobj]-> condition
    _HEDGE_ADJ = {"sure", "certain", "confident", "aware", "convinced"}
    for token in doc:
        if token.lemma_.lower() in _HEDGE_ADJ:
            for child in token.children:
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            for chunk in doc.noun_chunks:
                                if grandchild in chunk:
                                    return chunk.text.lower()

    # Priority 1: direct/prepositional object of a hedge/verification verb
    # e.g. "verify the equation" → "equation"
    keywords = set(_HEDGE_KEYWORDS + _VERIFICATION_KEYWORDS)
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() in keywords:
            for child in token.children:
                if child.dep_ in ("dobj", "pobj", "attr"):
                    for chunk in doc.noun_chunks:
                        if child in chunk:
                            return chunk.text.lower()

    # Priority 1.5: keyword-proximate noun chunk
    # Find where the triggering keyword sits in the token stream, then return
    # the first non-pronoun noun chunk that starts within 3 tokens after it,
    # PROVIDED no VERB/AUX token intervenes between keyword and chunk.
    #
    # "maybe a dash of symbolism"  → "a dash of symbolism"  ✓ (no verb gap)
    # "are typically harvested…"   → falls through to P2    ✓ (harvested is VERB)
    # "My process, I think, should be…" → falls through     ✓ (should/be are AUX/VERB)
    if matched_keyword:
        kw_words = matched_keyword.lower().split()
        kw_end_idx: Optional[int] = None
        for i in range(len(doc) - len(kw_words) + 1):
            if all(doc[i + j].text.lower() == kw_words[j] for j in range(len(kw_words))):
                kw_end_idx = i + len(kw_words)
                break

        if kw_end_idx is not None:
            verb_blocked = False   # True if a chunk was found but rejected due to verb gap
            for chunk in doc.noun_chunks:
                if kw_end_idx <= chunk.start <= kw_end_idx + 3:
                    if chunk.root.pos_ != "PRON":
                        # Reject the chunk if any VERB/AUX sits between keyword and chunk
                        has_verb_gap = any(
                            doc[k].pos_ in ("VERB", "AUX")
                            for k in range(kw_end_idx, chunk.start)
                        )
                        if not has_verb_gap:
                            return chunk.text.lower()
                        verb_blocked = True

            # Text fallback: only when NO chunk was found at all (not when verb-blocked).
            # Grab up to 2 noun/adjective tokens immediately after the keyword.
            if not verb_blocked:
                content = [
                    doc[k]
                    for k in range(kw_end_idx, min(kw_end_idx + 5, len(doc)))
                    if not doc[k].is_punct and not doc[k].is_space
                       and doc[k].pos_ in ("NOUN", "PROPN", "ADJ")
                ]
                if content:
                    return " ".join(t.text.lower() for t in content[:2])

    # Priority 2: non-pronoun nominal subject of the root verb
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.dep_ == "ROOT":
            if token.pos_ != "PRON":   # skip 'I', 'it', 'they'
                for chunk in doc.noun_chunks:
                    if token in chunk:
                        return chunk.text.lower()

    # Priority 3: first non-pronoun noun chunk
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ != "PRON":
            return chunk.text.lower()

    # Priority 4: regex fallback
    return _extract_subject_regex(sentence)


def _extract_subject_regex(sentence: str) -> str:
    """
    Lightweight regex/heuristic fallback when spaCy is unavailable.

    Strips common hedge/verification verbs and first-person pronouns,
    then returns the first meaningful noun phrase found.
    """
    text = sentence.lower().strip()

    # Pattern 0: "(I'm / I am) not sure about X" / "uncertain about X" → X
    m = re.search(
        r"(?:not\s+sure|uncertain|unsure|not\s+certain)\s+about\s+(.+?)(?:\.|,|;|$)",
        text, flags=re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    # Remove leading first-person / hedge opener phrases
    cleaned = re.sub(
        r"^(i think|i believe|let me|maybe|perhaps|possibly|probably|"
        r"i'm not sure|i am not sure|it seems|it appears|however,?)\s+",
        "", text, flags=re.IGNORECASE
    )
    # Remove auxiliary + verb at start (e.g. "verify the X" → "the X")
    cleaned = re.sub(
        r"^(verify|check|confirm|recalculate|validate|test|review|"
        r"inspect|determine|conclude)\s+(the\s+|that\s+|if\s+)?",
        "", cleaned, flags=re.IGNORECASE
    )
    # Take up to the first stop word / punctuation
    match = re.match(
        r"([a-z][a-z0-9 \-]*?)(?:\s+(?:is|are|was|were|might|could|seems|appears|will|has|have)\b|[,\.;:]|$)",
        cleaned
    )
    if match:
        subject = match.group(1).strip()
        if subject:
            return subject
    # Ultimate fallback: first 3 words
    words = cleaned.split()
    return " ".join(words[:3]) if words else sentence[:30]


def _extract_subject(sentence: str, nlp, matched_keyword: str = "") -> str:
    """Dispatch to spaCy or regex depending on whether spaCy loaded successfully."""
    if nlp is not None:
        return _extract_subject_spacy(sentence, nlp, matched_keyword=matched_keyword)
    return _extract_subject_regex(sentence)


# ---------------------------------------------------------------------------
# Step 4 — Embedding-based semantic similarity
# ---------------------------------------------------------------------------
def _get_embedding(text: str, model_name: str, model) -> np.ndarray:
    """Return a cached or freshly computed embedding for *text*."""
    cache_key = (model_name, text)
    if cache_key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[cache_key] = model.encode(text, convert_to_numpy=True)
    return _EMBEDDING_CACHE[cache_key]


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity clamped to [0.0, 1.0]."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    raw = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, raw))


def _compute_similarity(
    subj_a: str,
    subj_b: str,
    model_name: str,
    model,
) -> float:
    """Compute embedding-based cosine similarity between two subject strings."""
    if not subj_a or not subj_b:
        return 0.0
    emb_a = _get_embedding(subj_a, model_name, model)
    emb_b = _get_embedding(subj_b, model_name, model)
    return _cosine_similarity(emb_a, emb_b)


# ---------------------------------------------------------------------------
# Step 5 — Proximity score
# ---------------------------------------------------------------------------
def _proximity_score(gap: int, decay: float = 50.0) -> float:
    """Exponential decay: larger gap → smaller score. Result is in (0, 1]."""
    return math.exp(-gap / decay)


# ---------------------------------------------------------------------------
# Step 6 — Match score
# ---------------------------------------------------------------------------
def _match_score(
    subject_similarity: float,
    proximity: float,
    weights: Tuple[float, float] = (0.7, 0.3),
) -> float:
    """Weighted combination of semantic similarity and proximity."""
    w_sim, w_prox = weights
    return w_sim * subject_similarity + w_prox * proximity


# ---------------------------------------------------------------------------
# Step 7 — Resolution assignment
# ---------------------------------------------------------------------------
def _assign_resolutions(
    hedges: List[_Statement],
    verifications: List[_Statement],
    model_name: str,
    model,
    threshold: float,
    decay: float,
    weights: Tuple[float, float],
) -> List[_HedgeResult]:
    """
    For each hedge, evaluate all *future* verifications and select the best match.

    A verification may resolve multiple hedges only if subject_similarity > 0.90
    for every hedge it resolves (per spec §7).
    """
    results: List[_HedgeResult] = []

    # Track which verifications are already claimed exclusively
    # (i.e., similarity ≤ 0.90 for this hedge but already matched to another)
    exclusively_claimed: Dict[str, float] = {}  # ver_id -> best similarity used

    for hedge in hedges:
        best_score = -1.0
        best_ver: Optional[_Statement] = None
        best_sim = 0.0
        best_prox = 0.0

        for ver in verifications:
            # Only consider verifications that appear *after* the hedge
            if ver.position <= hedge.position:
                continue

            sim = _compute_similarity(hedge.subject, ver.subject, model_name, model)
            gap = ver.position - hedge.position
            prox = _proximity_score(gap, decay)
            score = _match_score(sim, prox, weights)

            # If verification is exclusively claimed, only allow re-use when
            # subject_similarity > 0.90 for this hedge too.
            if ver.id in exclusively_claimed:
                if sim <= 0.90:
                    continue

            if score > best_score:
                best_score = score
                best_ver = ver
                best_sim = sim
                best_prox = prox

        resolved = (best_ver is not None) and (best_score >= threshold)

        result = _HedgeResult(
            id=hedge.id,
            subject=hedge.subject,
            position=hedge.position,
            sentence=hedge.sentence,
            matched_keyword=getattr(hedge, "_matched_keyword", ""),
        )

        if best_ver is not None:
            result.matched_verification = best_ver.id if resolved else None
            result.subject_similarity = round(best_sim, 4)
            result.proximity_score = round(best_prox, 4)
            result.match_score = round(best_score, 4)
            result.resolved = resolved

            # Mark this verification as claimed; record the similarity
            if resolved and best_sim <= 0.90:
                exclusively_claimed[best_ver.id] = best_sim
        else:
            result.matched_verification = None
            result.resolved = False

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Step 8 — Uncertainty weight adjustment
# ---------------------------------------------------------------------------
def _adjust_weights(
    results: List[_HedgeResult],
    reduction: float = 0.8,
) -> None:
    """Mutate each _HedgeResult in place, computing effective_weight."""
    for r in results:
        if r.resolved and r.match_score is not None:
            r.effective_weight = round(1.0 * (1.0 - reduction * r.match_score), 4)
        else:
            r.effective_weight = 1.0


# ---------------------------------------------------------------------------
# Step 9 helpers — Certainty classification signals
# ---------------------------------------------------------------------------

# Phrases strongly associated with a definitive conclusion
_FINALITY_CERTAIN: List[str] = [
    "therefore", "thus", "hence", "consequently", "as a result",
    "i can conclude", "the answer is", "the conclusion is",
    "with confidence", "can be stated", "has been confirmed",
    "is confirmed", "is verified", "without doubt",
    "definitively", "unambiguously", "can be stated with confidence",
    "the evidence fully supports", "fully supports",
]

# Phrases that indicate lingering uncertainty at the conclusion
_FINALITY_UNCERTAIN: List[str] = [
    "still not sure", "still uncertain", "remains unclear",
    "cannot determine", "cannot conclude", "unable to confirm",
    "i'm not sure", "i am not sure", "unclear", "ambiguous",
    "further investigation", "more information needed",
    "insufficient evidence", "inconclusive", "remains open",
]


def _late_unresolved_ratio(results: List[_HedgeResult], total_sentences: int) -> float:
    """
    Fraction of *unresolved* hedges whose sentence position falls in the
    second half of the document.

    High value → uncertainty persists into the conclusion (signals Uncertain).
    Low value  → remaining doubt was expressed early and not carried forward.
    """
    unresolved = [r for r in results if not r.resolved]
    if not unresolved or total_sentences == 0:
        return 0.0
    midpoint = total_sentences / 2.0
    late_count = sum(1 for r in unresolved if r.position >= midpoint)
    return round(late_count / len(unresolved), 4)


def _conclusion_finality(sentences: List[str], window: int = 3) -> float:
    """
    Scan the last *window* sentences for definitive vs uncertain language.

    Returns a score in [-1.0, +1.0]:
      > 0  → conclusion leans definitive (supports Certain)
      < 0  → conclusion leans uncertain
      = 0  → neutral / no signal
    """
    tail = sentences[-window:] if len(sentences) >= window else sentences
    tail_text = " ".join(tail).lower()

    certain_hits  = sum(1 for p in _FINALITY_CERTAIN   if p in tail_text)
    uncertain_hits = sum(1 for p in _FINALITY_UNCERTAIN if p in tail_text)

    total = certain_hits + uncertain_hits
    if total == 0:
        return 0.0
    return round((certain_hits - uncertain_hits) / total, 4)


def _predict_certainty(
    total_hedges: int,
    trur: float,
    late_unresolved: float,
    finality: float,
    hvr: float,
) -> str:
    """
    Combine TRUR, HVR, late-unresolved ratio and conclusion finality into a
    binary certainty label: ``"Certain"`` or ``"Uncertain"``.

    Composite score (0 → 1, higher = more certain):
      45 % — HVR certainty  1 / (1 + HVR); HVR = hedges / verifications
                            low ratio (few hedges vs many verifications) → Certain
                            high ratio (many hedges vs few verifications) → Uncertain
      25 % — Position       fraction of unresolved hedges in the first half (early = Certain)
      15 % — TRUR           resolution rate (resolved / total hedges)
      15 % — Finality       definitive vs uncertain conclusion language

    Fixes applied
    ─────────────
    Fix 1 — Hedge density guard:
      With fewer than 3 detected hedges there is insufficient signal to
      predict "Certain" reliably — force "Uncertain".

    Fix 2 — Finality gate:
      When no hedge was resolved (TRUR = 0), conclusion language alone cannot
      push the prediction to "Certain". Finality weight drops 15 % → 5 % and
      the freed weight redistributes to the position signal.

    Threshold:
      score ≥ 0.50 → "Certain"
      score  < 0.50 → "Uncertain"
    """
    # Fix 1: insufficient hedge count — force Uncertain
    if total_hedges < 3:
        return "Uncertain"

    # Fix 2: gate finality weight on whether any hedge was actually resolved
    if trur > 0:
        finality_weight = 0.15
        position_weight = 0.25
    else:
        # No resolution — demote finality, promote position signal
        finality_weight = 0.05
        position_weight = 0.35

    hvr_certainty = 1.0 / (1.0 + hvr)   # maps HVR [0, ∞) → (0, 1]

    score = (
        trur                      * 0.15
        + hvr_certainty           * 0.45
        + (1.0 - late_unresolved) * position_weight
        + ((finality + 1.0) / 2.0) * finality_weight   # map [-1,1] → [0,1]
    )

    return "Certain" if score >= 0.50 else "Uncertain"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    """
    Split a reasoning trace into sentences.

    Handles common sentence-ending punctuation and newline-delimited blocks
    that reasoning models often produce.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on sentence-terminating punctuation OR on blank lines
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    # Further split long lines that lack terminal punctuation but contain newlines
    sentences: List[str] = []
    for segment in raw:
        lines = [l.strip() for l in segment.split("\n") if l.strip()]
        sentences.extend(lines)
    return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def calculate_proximity_metrics(
    reasoning_text: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.60,
    proximity_decay: float = 50.0,
    match_weights: Tuple[float, float] = (0.7, 0.3),
    resolution_weight_reduction: float = 0.8,
) -> dict:
    """
    Analyse a reasoning trace and return Topic-Aware Self-Correction Proximity metrics.

    Parameters
    ----------
    reasoning_text : str
        The raw reasoning / chain-of-thought text produced by the LLM.
    embedding_model : str
        SentenceTransformer model name (default: ``all-MiniLM-L6-v2``).
        Override via the ``PROXIMITY_EMBEDDING_MODEL`` environment variable.
    similarity_threshold : float
        Minimum match_score for a hedge to be marked as resolved (default: 0.60).
    proximity_decay : float
        Decay constant for the exponential proximity formula (default: 50.0).
    match_weights : tuple[float, float]
        Weights for (subject_similarity, proximity_score) in match_score (default: 0.7, 0.3).
    resolution_weight_reduction : float
        Fraction of match_score subtracted from hedge_weight when resolved (default: 0.8).

    Returns
    -------
    dict with keys:
        total_hedges, resolved_hedges, unresolved_hedges, trur,
        weighted_trur, matches
    """
    # Allow .env override for the embedding model
    embedding_model = os.getenv("PROXIMITY_EMBEDDING_MODEL", embedding_model)

    if not reasoning_text or not reasoning_text.strip():
        logger.info("[proximity_metric] Empty reasoning text — returning zero metrics.")
        return {
            "total_hedges": 0,
            "resolved_hedges": 0,
            "unresolved_hedges": 0,
            "trur": 0.0,
            "weighted_trur": 0.0,
            "matches": [],
        }

    # --- Load singletons ---
    model = _load_embedding_model(embedding_model)
    nlp   = _load_nlp()

    # --- Sentence segmentation ---
    sentences = _split_sentences(reasoning_text)
    logger.debug("[proximity_metric] Segmented %d sentences.", len(sentences))

    # --- Steps 1 & 2: Detect hedges and verifications ---
    hedges        = _detect_hedges(sentences)
    verifications = _detect_verifications(sentences)

    logger.info(
        "[proximity_metric] Found %d hedge(s) and %d verification(s).",
        len(hedges), len(verifications),
    )

    if not hedges:
        # No hedge case — classify as Certain (no expressed uncertainty detected)
        logger.info("[proximity_metric] No hedges detected — classifying as Certain.")
        total_vers = len(verifications)
        finality   = _conclusion_finality(sentences)
        return {
            "total_hedges":          0,
            "resolved_hedges":       0,
            "unresolved_hedges":     0,
            "trur":                  0.0,
            "weighted_trur":         0.0,
            "total_verifications":   total_vers,
            "hvr":                   0.0,
            "late_unresolved_ratio": 0.0,
            "conclusion_finality":   round(finality, 4),
            "predicted_certainty":   "Certain",
            "matches":               [],
        }

    # --- Step 3: Extract subjects ---
    for stmt in hedges:
        kw = getattr(stmt, "_matched_keyword", "")
        stmt.subject = _extract_subject(stmt.sentence, nlp, matched_keyword=kw)
    for stmt in verifications:
        stmt.subject = _extract_subject(stmt.sentence, nlp)

    # --- Steps 4-7: Assign resolutions ---
    results = _assign_resolutions(
        hedges, verifications,
        model_name=embedding_model,
        model=model,
        threshold=similarity_threshold,
        decay=proximity_decay,
        weights=match_weights,
    )

    # --- Step 8: Adjust weights ---
    _adjust_weights(results, reduction=resolution_weight_reduction)

    # --- Aggregate statistics ---
    total      = len(results)
    resolved   = sum(1 for r in results if r.resolved)
    unresolved = total - resolved
    trur       = round(resolved / total, 4) if total > 0 else 0.0

    # weighted_trur: mean fraction of uncertainty eliminated across all hedges
    weight_reductions = [round(1.0 - r.effective_weight, 4) for r in results]
    weighted_trur     = round(sum(weight_reductions) / total, 4) if total > 0 else 0.0

    matches = []
    for r in results:
        record: dict = {
            "id":                    r.id,
            "sentence":              r.sentence,
            "matched_keyword":       r.matched_keyword,
            "subject":               r.subject,
            "position":              r.position,
            "matched_verification":  r.matched_verification,
            "subject_similarity":    r.subject_similarity,
            "proximity_score":       r.proximity_score,
            "match_score":           r.match_score,
            "resolved":              r.resolved,
            "effective_weight":      r.effective_weight,
        }
        matches.append(record)

    # --- Step 9: Certainty classification ---
    total_vers       = len(verifications)
    hvr              = round(total / max(1, total_vers), 4)   # hedge-to-verify ratio
    total_sents      = len(sentences)
    late_unresolved  = _late_unresolved_ratio(results, total_sents)
    finality         = _conclusion_finality(sentences)
    predicted        = _predict_certainty(total, trur, late_unresolved, finality, hvr)

    return {
        "total_hedges":          total,
        "resolved_hedges":       resolved,
        "unresolved_hedges":     unresolved,
        "trur":                  trur,
        "weighted_trur":         weighted_trur,
        "total_verifications":   total_vers,
        "hvr":                   hvr,
        "late_unresolved_ratio": late_unresolved,
        "conclusion_finality":   round(finality, 4),
        "predicted_certainty":   predicted,
        "matches":               matches,
    }


# ---------------------------------------------------------------------------
# Smoke-test / CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    SAMPLE_TRACE = """
Maybe the equation is incorrect.
Let me verify the equation.
The equation is correct.

However, I'm not sure about the boundary condition.
The final answer seems reasonable.
"""

    print("\n" + "=" * 60)
    print("PROXIMITY METRIC — SMOKE TEST")
    print("=" * 60)
    print("Input reasoning trace:")
    print(SAMPLE_TRACE)

    result = calculate_proximity_metrics(SAMPLE_TRACE)

    print("\nResult:")
    print(json.dumps(result, indent=2))

    print("\nSummary:")
    print(f"  Total hedges     : {result['total_hedges']}")
    print(f"  Resolved hedges  : {result['resolved_hedges']}")
    print(f"  Unresolved hedges: {result['unresolved_hedges']}")
    print(f"  TRUR             : {result['trur']:.2%}")
    print(f"  Weighted TRUR    : {result['weighted_trur']:.2%}")

    # Exit with non-zero if no hedges detected (likely an env issue)
    if result["total_hedges"] == 0:
        print("\n[WARN] No hedges detected — check that input text is non-empty.", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
