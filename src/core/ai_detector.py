"""
ai_detector.py
--------------
Detects AI-generated text using a RoBERTa-based classifier from HuggingFace.

Model: roberta-base-openai-detector
  - Lightweight RoBERTa model fine-tuned to distinguish AI-generated text
  - Returns probability scores for AI-generated content
  - Efficient for batch processing of document chunks
"""

import os
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Singleton model loader ─────────────────────────────────────────────────────
# We load the model once and reuse it across calls to avoid repeated I/O.
_DEFAULT_MODEL_NAME = "roberta-base-openai-detector"
_model: AutoModelForSequenceClassification | None = None
_tokenizer: AutoTokenizer | None = None


def _get_model_name() -> str:
    """Return the configured AI detection model name."""
    return os.getenv("AI_DETECTION_MODEL", _DEFAULT_MODEL_NAME)


def _get_model_and_tokenizer() -> (
    tuple[AutoModelForSequenceClassification, AutoTokenizer]
):
    """Lazy-load the AI detection model and tokenizer (singleton pattern)."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        model_name = _get_model_name()
        print(f"[ai_detector] Loading model: {model_name} …")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("[ai_detector] Model loaded successfully.")

        # Move to GPU if available for faster inference
        if torch.cuda.is_available():
            _model = _model.to("cuda")
            print("[ai_detector] Using GPU for inference.")
    return _model, _tokenizer


# ── Public API ─────────────────────────────────────────────────────────────────


def detect_ai_probability(text: str) -> float:
    """
    Detect the probability that a given text was AI-generated.

    Args:
        text: Input text string to analyze.

    Returns:
        Probability score between 0.0 (human-written) and 1.0 (AI-generated).
    """
    if not text or not text.strip():
        return 0.0

    model, tokenizer = _get_model_and_tokenizer()

    # Tokenize input
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Assuming the model outputs [human, AI] or similar
        # We'll take the AI class probability (usually index 1)
        ai_prob = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()

    return float(ai_prob)


def detect_ai_probability_batch(texts: List[str], batch_size: int = 8) -> List[float]:
    """
    Detect AI probability for multiple texts in batch for efficiency.

    Args:
        texts: List of text strings to analyze.
        batch_size: Number of texts to process per batch.

    Returns:
        List of probability scores (0.0 to 1.0) corresponding to input texts.
    """
    if not texts:
        return []

    model, tokenizer = _get_model_and_tokenizer()
    probabilities = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Filter empty texts
        valid_texts = [t for t in batch_texts if t and t.strip()]
        valid_indices = [idx for idx, t in enumerate(batch_texts) if t and t.strip()]

        if not valid_texts:
            probabilities.extend([0.0] * len(batch_texts))
            continue

        # Tokenize batch
        inputs = tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Extract AI probabilities
            batch_probs = []
            for j in range(probs.shape[0]):
                ai_prob = (
                    probs[j, 1].item() if probs.shape[1] > 1 else probs[j, 0].item()
                )
                batch_probs.append(float(ai_prob))

        # Map back to original batch order
        batch_result = [0.0] * len(batch_texts)
        for idx, prob in zip(valid_indices, batch_probs):
            batch_result[idx] = prob

        probabilities.extend(batch_result)

    return probabilities


def detect_document_ai_probability(chunks: List[str]) -> Dict[str, float]:
    """
    Detect AI probability for a document by analyzing its chunks.

    Args:
        chunks: List of text chunks from a document.

    Returns:
        Dict with keys:
          - 'overall': Mean AI probability across all chunks
          - 'max': Maximum AI probability across all chunks
          - 'chunk_scores': List of per-chunk AI probabilities
    """
    if not chunks:
        return {"overall": 0.0, "max": 0.0, "chunk_scores": []}

    chunk_scores = detect_ai_probability_batch(chunks)

    return {
        "overall": float(np.mean(chunk_scores)) if chunk_scores else 0.0,
        "max": float(np.max(chunk_scores)) if chunk_scores else 0.0,
        "chunk_scores": chunk_scores,
    }


def detect_documents_ai_probability(
    chunked_docs: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Detect AI probability for multiple documents.

    Args:
        chunked_docs: Dict mapping document name → list of chunk strings.

    Returns:
        Dict mapping document name → AI probability dict (overall, max, chunk_scores).
    """
    results = {}

    for doc_name, chunks in chunked_docs.items():
        results[doc_name] = detect_document_ai_probability(chunks)

    return results
