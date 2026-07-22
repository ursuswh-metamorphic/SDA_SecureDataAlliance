"""normalize.py — shared text/ID normalization for the P3 clean protocol.

Single source of truth for how doc names and page texts are normalized
BEFORE hashing into canonical passage IDs. Any change here MUST bump
NORMALIZE_VERSION, which is baked into every passage_id — so old and new
manifests can never silently mix.

Used by: build_corpus_manifest.py, build_qrels.py, audit_protocol.py.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

# Baked into every hash. Bump on ANY change to norm_text/norm_doc_name.
NORMALIZE_VERSION = "v1"

_WS_RE = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """NFC unicode + collapse all whitespace runs to single spaces + strip.

    Case is PRESERVED for passage text (case can be meaningful in financial
    filings); only doc names are lowercased (norm_doc_name).
    """
    s = unicodedata.normalize("NFC", str(s))
    return _WS_RE.sub(" ", s).strip()


def norm_doc_name(s: str) -> str:
    """NFC + collapse whitespace + lowercase — doc names are case-insensitive."""
    return norm_text(s).lower()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def passage_id(doc_name: str, page_num: int | str, text: str) -> str:
    """Canonical stable passage ID (24 hex chars).

    sha256('<ver>|' + norm(doc_name) + '|' + page_num + '|' + norm(text))[:24]
    Independent of JSON ordering / file layout. NOT derived from the
    original dataset's incremental 'index' field.
    """
    key = f"{NORMALIZE_VERSION}|{norm_doc_name(doc_name)}|{page_num}|{norm_text(text)}"
    return sha256_hex(key)[:24]


def parent_document_id(doc_name: str) -> str:
    """Stable parent document ID (16 hex chars)."""
    key = f"{NORMALIZE_VERSION}|{norm_doc_name(doc_name)}"
    return sha256_hex(key)[:16]


def text_fingerprint(text: str) -> str:
    """SHA-256 of the NORMALIZED text — used for exact-duplicate detection."""
    return sha256_hex(norm_text(text))
