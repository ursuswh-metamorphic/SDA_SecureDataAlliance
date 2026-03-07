import base64
import logging
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import Node, NodeWithScore, QueryBundle, TextNode

LOGGER = logging.getLogger(__name__)


def _lazy_import_presidio():
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerResult
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
        return AnalyzerEngine, RecognizerResult, NlpEngineProvider, AnonymizerEngine
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.warning("Presidio is not fully available: %s", exc)
        return None, None, None, None


def _lazy_import_tenseal():
    try:
        import tenseal as ts

        return ts
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.warning("TenSEAL is not available: %s", exc)
        return None


def _weighted_average_manual(weights_results: List[Tuple[List[np.ndarray], float]]) -> List[np.ndarray]:
    """
    Manual implementation of weighted_average for Flower compatibility.
    
    Args:
        weights_results: List of tuples (parameters, weight) where parameters is a list of numpy arrays
        
    Returns:
        List of weighted averaged numpy arrays
    """
    if not weights_results:
        return []
    
    # Get the structure from first element
    num_arrays = len(weights_results[0][0])
    total_weights = sum(weight for _, weight in weights_results)
    
    if total_weights == 0:
        return [np.zeros_like(weights_results[0][0][i]) for i in range(num_arrays)]
    
    # Initialize result arrays
    result = []
    for i in range(num_arrays):
        # Get shape from first array
        shape = weights_results[0][0][i].shape
        weighted_sum = np.zeros(shape, dtype=np.float64)
        
        # Sum weighted arrays
        for params, weight in weights_results:
            weighted_sum += params[i] * weight
        
        # Divide by total weight
        result.append(weighted_sum / total_weights)
    
    return result


def _lazy_import_flower():
    """
    Try to import weighted_average from Flower, fallback to manual implementation.
    """
    try:
        from flwr.common import weighted_average
        LOGGER.info("Using Flower's weighted_average")
        return weighted_average
    except ImportError:
        try:
            # Try alternative location in newer Flower versions
            from flwr.server.strategy import weighted_average
            LOGGER.info("Using Flower's weighted_average from flwr.server.strategy")
            return weighted_average
        except ImportError:
            # Fallback to manual implementation
            LOGGER.info("Flower weighted_average not found, using manual implementation")
            return _weighted_average_manual
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.warning("Flower is not available: %s, using manual implementation", exc)
        return _weighted_average_manual


@dataclass
class PrivacySummaryConfig:
    """Configuration for the privacy-aware summarization postprocessor."""

    enable: bool = True
    language: str = "en"
    allowed_entities: List[str] = field(
        default_factory=lambda: [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "CREDIT_CARD",
            "LOCATION",
            "US_SSN",
        ]
    )
    risk_threshold: float = 0.45
    drop_high_risk_sentences: bool = True
    preserve_sentence_count: int = 1
    high_risk_keywords: List[str] = field(
        default_factory=lambda: [
            "password",
            "secret",
            "confidential",
            "ssn",
            "social security",
            "credit card",
            "bank account",
            "private",
            "classified",
        ]
    )
    anonymizer_policy: str = "replace"
    anonymizer_placeholder: str = "<REDACTED>"
    encryption_enabled: bool = True
    encryption_chunk_size: int = 256
    encryption_poly_modulus_degree: int = 8192
    encryption_coeff_mod_bit_sizes: Optional[List[int]] = field(
        default_factory=lambda: [60, 40, 40, 60]
    )
    encryption_global_scale: float = 2**40
    federated_enabled: bool = False
    federated_min_threshold: float = 0.25
    federated_max_threshold: float = 0.85
    federated_weight: float = 1.0
    passthrough_entities: List[str] = field(default_factory=list)
    log_stats: bool = True
    # Mask all detected entities with entity type tags (e.g., <MONEY>, <DATE>)
    mask_all_entities: bool = False
    use_entity_type_tags: bool = True  # Use <ENTITY_TYPE> instead of <REDACTED>

    @classmethod
    def from_dict(cls, raw_cfg: Optional[Dict[str, Any]]) -> "PrivacySummaryConfig":
        if raw_cfg is None:
            return cls()

        mapping = {
            "enable_privacy_summary": "enable",
            "presidio_language": "language",
            "presidio_entities": "allowed_entities",
            "eraser_risk_threshold": "risk_threshold",
            "eraser_drop_high_risk": "drop_high_risk_sentences",
            "eraser_preserve_sentences": "preserve_sentence_count",
            "eraser_keywords": "high_risk_keywords",
            "tenseal_enabled": "encryption_enabled",
            "tenseal_chunk_size": "encryption_chunk_size",
            "tenseal_poly_modulus_degree": "encryption_poly_modulus_degree",
            "tenseal_coeff_mod_bit_sizes": "encryption_coeff_mod_bit_sizes",
            "tenseal_global_scale": "encryption_global_scale",
            "flower_enabled": "federated_enabled",
            "flower_threshold_bounds": None,
            "flower_weight": "federated_weight",
            "privacy_log_stats": "log_stats",
            "privacy_passthrough_entities": "passthrough_entities",
            "mask_all_entities": "mask_all_entities",
            "use_entity_type_tags": "use_entity_type_tags",
        }

        cfg_dict = {}
        for key, value in raw_cfg.items():
            if key in mapping and mapping[key] is not None:
                cfg_dict[mapping[key]] = value
            elif key == "flower_threshold_bounds" and isinstance(value, (list, tuple)):
                if len(value) == 2:
                    cfg_dict["federated_min_threshold"] = float(value[0])
                    cfg_dict["federated_max_threshold"] = float(value[1])

        return cls(**cfg_dict)


class PresidioSanitizer:
    """Wrapper around Presidio with graceful fallbacks."""

    FALLBACK_PATTERNS = {
        "EMAIL_ADDRESS": re.compile(
            r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", re.IGNORECASE
        ),
        "PHONE_NUMBER": re.compile(r"(\+?\d[\d\s-]{6,}\d)"),
        "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(
            r"\b(?:\d[ -]*?){13,16}\b"
        ),
    }

    def __init__(self, cfg: PrivacySummaryConfig):
        self.cfg = cfg
        (
            AnalyzerEngine,
            RecognizerResult,
            NlpEngineProvider,
            AnonymizerEngine,
        ) = _lazy_import_presidio()
        self._analyzer_cls = AnalyzerEngine
        self._recognizer_result_cls = RecognizerResult
        self._nlp_provider_cls = NlpEngineProvider
        self._anonymizer_cls = AnonymizerEngine
        self.analyzer = None
        self.anonymizer = None
        self.available = False
        self._init_engines()

    def _init_engines(self) -> None:
        if (
            self._analyzer_cls is None
            or self._nlp_provider_cls is None
            or self._anonymizer_cls is None
        ):
            return

        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {
                    "lang_code": self.cfg.language,
                    "model_name": f"{self.cfg.language}_core_web_sm",
                }
            ],
        }

        try:
            provider = self._nlp_provider_cls(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning(
                "Failed to initialize Presidio NLP engine (language=%s): %s",
                self.cfg.language,
                exc,
            )
            nlp_engine = None

        try:
            if nlp_engine is not None:
                self.analyzer = self._analyzer_cls(
                    nlp_engine=nlp_engine, supported_languages=[self.cfg.language]
                )
            else:
                self.analyzer = self._analyzer_cls()
            self.anonymizer = self._anonymizer_cls()
            self.available = True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Presidio initialization failed: %s", exc)
            self.available = False

    def analyze(self, text: str) -> Dict[str, Any]:
        raw_results: Iterable[Any] = []
        normalized: List[Dict[str, Any]] = []

        if text.strip() == "":
            return {"results": [], "normalized": []}

        if self.available and self.analyzer is not None:
            try:
                # If mask_all_entities is True, detect ALL entities (not just allowed_entities)
                entities_to_detect = None if self.cfg.mask_all_entities else (self.cfg.allowed_entities or None)
                
                raw_results = self.analyzer.analyze(
                    text=text,
                    language=self.cfg.language,
                    entities=entities_to_detect,
                    return_decision_process=False,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning("Presidio analyze failed, falling back to regex: %s", exc)
                raw_results = []

        normalized = self._normalize_results(raw_results)
        if not normalized:
            normalized = self._regex_detect(text)

        return {"results": list(raw_results), "normalized": normalized}

    def anonymize(
        self, text: str, analysis: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if text.strip() == "":
            return text, []

        raw_results = analysis.get("results", [])
        normalized = analysis.get("normalized", [])

        if self.available and self.anonymizer is not None and raw_results:
            operators: Dict[str, Dict[str, Any]] = {}
            placeholder_template = (
                self.cfg.anonymizer_placeholder
                if self.cfg.anonymizer_policy == "replace"
                else None
            )
            
            # Build operators dict
            for res in raw_results:
                # Handle both RecognizerResult objects and dicts
                if isinstance(res, dict):
                    entity_type = res.get("entity_type", "DEFAULT")
                else:
                    entity_type = getattr(res, "entity_type", "DEFAULT")
                
                if entity_type in self.cfg.passthrough_entities:
                    continue
                
                # Use entity type tags if enabled, otherwise use placeholder
                if self.cfg.use_entity_type_tags:
                    placeholder = f"<{entity_type}>"
                elif placeholder_template:
                    placeholder = placeholder_template
                else:
                    placeholder = f"<{entity_type}>"
                
                operators[entity_type] = {"type": "replace", "new_value": placeholder}

            try:
                # Ensure raw_results is a list of RecognizerResult objects, not dicts
                analyzer_results = []
                for res in raw_results:
                    if isinstance(res, dict):
                        # Convert dict to RecognizerResult if possible
                        if self._recognizer_result_cls is not None:
                            try:
                                analyzer_results.append(
                                    self._recognizer_result_cls(
                                        entity_type=res.get("entity_type", "DEFAULT"),
                                        start=res.get("start", 0),
                                        end=res.get("end", 0),
                                        score=res.get("score", 0.0)
                                    )
                                )
                            except Exception:
                                # If conversion fails, skip this result
                                continue
                        else:
                            continue
                    else:
                        analyzer_results.append(res)
                
                if not analyzer_results:
                    # If no valid results, use fallback
                    return self._fallback_anonymize(text, normalized)
                
                result = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results,
                    operators=operators or {
                        "DEFAULT": {
                            "type": "replace",
                            "new_value": self.cfg.anonymizer_placeholder,
                        }
                    },
                )
                return result.text, normalized
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning("Presidio anonymize failed, using fallback: %s", exc)

        return self._fallback_anonymize(text, normalized)

    def _normalize_results(self, results: Iterable[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for res in results or []:
            entity_type = getattr(res, "entity_type", None)
            start = getattr(res, "start", None)
            end = getattr(res, "end", None)
            score = getattr(res, "score", None)
            if entity_type is None or start is None or end is None:
                continue
            normalized.append(
                {
                    "entity_type": entity_type,
                    "start": int(start),
                    "end": int(end),
                    "score": float(score) if score is not None else None,
                }
            )
        return normalized

    def _regex_detect(self, text: str) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for entity, pattern in self.FALLBACK_PATTERNS.items():
            for match in pattern.finditer(text):
                normalized.append(
                    {
                        "entity_type": entity,
                        "start": match.start(),
                        "end": match.end(),
                        "score": None,
                    }
                )
        return normalized

    def _fallback_anonymize(
        self, text: str, results: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not results:
            return text, []
        masked_text = text
        offset = 0
        replacements: List[Dict[str, Any]] = []
        for res in sorted(results, key=lambda item: item["start"]):
            if res["entity_type"] in self.cfg.passthrough_entities:
                continue
            start = res["start"] + offset
            end = res["end"] + offset
            
            # Use entity type tags if enabled, otherwise use placeholder
            if self.cfg.use_entity_type_tags:
                placeholder = f"<{res['entity_type']}>"
            else:
                placeholder = self.cfg.anonymizer_placeholder
            
            masked_text = (
                masked_text[:start] + placeholder + masked_text[end:]
            )
            offset += len(placeholder) - (res["end"] - res["start"])
            replacements.append(res)
        return masked_text, replacements


class Eraser4RAGSanitizer:
    """Context-aware sanitization inspired by Eraser4RAG."""

    SENTENCE_REGEX = re.compile(r"(?<=[.!?\n])\s+")

    def __init__(self, cfg: PrivacySummaryConfig):
        self.cfg = cfg
        self.threshold = float(max(0.0, min(1.0, cfg.risk_threshold)))

    def update_threshold(self, new_threshold: float) -> None:
        self.threshold = float(max(0.0, min(1.0, new_threshold)))

    def sanitize(
        self,
        text: str,
        query: Optional[str],
        pii_spans: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        sentences = self._split_sentences(text)
        if not sentences:
            return text, {
                "removed_count": 0,
                "kept_count": 0,
                "average_risk": 0.0,
                "removed_sentences": [],
            }

        pii_spans_sorted = sorted(pii_spans, key=lambda item: item["start"])
        sanitized_sentences: List[Dict[str, Any]] = []
        removed_sentences: List[Dict[str, Any]] = []
        risk_scores: List[float] = []

        for idx, sentence in enumerate(sentences):
            risk = self._compute_risk(sentence, query, pii_spans_sorted)
            risk_scores.append(risk)
            sentence_payload = {
                "index": idx,
                "text": sentence["text"],
                "start": sentence["start"],
                "end": sentence["end"],
                "risk": risk,
            }
            if (
                self.cfg.drop_high_risk_sentences
                and risk >= self.threshold
                and len(sanitized_sentences) >= self.cfg.preserve_sentence_count
            ):
                removed_sentences.append(sentence_payload)
                continue
            sanitized_sentences.append(sentence_payload)

        if (
            self.cfg.preserve_sentence_count > 0
            and len(sanitized_sentences) < self.cfg.preserve_sentence_count
            and removed_sentences
        ):
            removed_sorted = sorted(removed_sentences, key=lambda item: item["risk"])
            needed = self.cfg.preserve_sentence_count - len(sanitized_sentences)
            sanitized_sentences.extend(removed_sorted[:needed])
            removed_sentences = removed_sorted[needed:]

        sanitized_sentences = sorted(sanitized_sentences, key=lambda item: item["index"])
        sanitized_text = " ".join(
            sentence["text"] for sentence in sanitized_sentences
        ).strip()

        metadata = {
            "removed_count": len(removed_sentences),
            "kept_count": len(sanitized_sentences),
            "average_risk": float(np.mean(risk_scores)) if risk_scores else 0.0,
            "max_risk": float(np.max(risk_scores)) if risk_scores else 0.0,
            "removed_sentences": removed_sentences,
        }
        return sanitized_text or text, metadata

    def _split_sentences(self, text: str) -> List[Dict[str, Any]]:
        sentences: List[Dict[str, Any]] = []
        if not text:
            return sentences
        cursor = 0
        for match in self.SENTENCE_REGEX.finditer(text):
            end = match.start()
            sentence = text[cursor:end].strip()
            if sentence:
                sentences.append(
                    {"text": sentence, "start": cursor, "end": end}
                )
            cursor = match.end()
        if cursor < len(text):
            tail = text[cursor:].strip()
            if tail:
                sentences.append(
                    {"text": tail, "start": cursor, "end": len(text)}
                )
        return sentences

    def _compute_risk(
        self,
        sentence: Dict[str, Any],
        query: Optional[str],
        pii_spans: List[Dict[str, Any]],
    ) -> float:
        text = sentence["text"]
        if not text:
            return 0.0

        length = max(1, sentence["end"] - sentence["start"])
        pii_overlap_chars = 0
        for span in pii_spans:
            overlap_start = max(sentence["start"], span["start"])
            overlap_end = min(sentence["end"], span["end"])
            if overlap_start < overlap_end:
                pii_overlap_chars += overlap_end - overlap_start

        pii_ratio = pii_overlap_chars / length
        keyword_score = 0.0
        lowered = text.lower()
        for keyword in self.cfg.high_risk_keywords:
            if keyword.lower() in lowered:
                keyword_score += 0.2

        numeric_score = min(0.3, self._numeric_density(text) * 0.3)
        query_score = 0.0
        if query:
            query_terms = {token.lower() for token in query.split() if token}
            if query_terms:
                overlap_terms = sum(
                    1 for token in text.split() if token.lower() in query_terms
                )
                query_score = min(0.2, overlap_terms / max(1, len(query_terms)))

        risk = min(1.0, pii_ratio + keyword_score + numeric_score + query_score)
        return float(risk)

    @staticmethod
    def _numeric_density(sentence: str) -> float:
        if not sentence:
            return 0.0
        digits = sum(ch.isdigit() for ch in sentence)
        return digits / max(1, len(sentence))


class TenSEALEncryptor:
    """Encrypt sanitized summaries using TenSEAL (CKKS)."""

    def __init__(self, cfg: PrivacySummaryConfig):
        self.cfg = cfg
        self.ts = _lazy_import_tenseal()
        self.available = self.ts is not None and cfg.encryption_enabled
        self.context = None
        if self.available:
            self._init_context()

    def _init_context(self) -> None:
        try:
            self.context = self.ts.context(
                self.ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.cfg.encryption_poly_modulus_degree,
                coeff_mod_bit_sizes=self.cfg.encryption_coeff_mod_bit_sizes,
            )
            self.context.generate_galois_keys()
            self.context.global_scale = self.cfg.encryption_global_scale
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Failed to initialize TenSEAL context: %s", exc)
            self.available = False
            self.context = None

    def encrypt(self, text: str) -> Optional[str]:
        if not self.available or self.context is None:
            return None
        if not text:
            return None

        vector = self._text_to_vector(text)
        try:
            encrypted = self.ts.ckks_vector(self.context, vector)
            serialized = encrypted.serialize()
            return base64.b64encode(serialized).decode("ascii")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("TenSEAL encryption failed: %s", exc)
            return None

    def _text_to_vector(self, text: str) -> List[float]:
        encoded = text.encode("utf-8")
        chunk_size = max(1, self.cfg.encryption_chunk_size)
        truncated = encoded[:chunk_size]
        vector = [float(byte) / 255.0 for byte in truncated]
        if len(vector) < chunk_size:
            vector.extend([0.0] * (chunk_size - len(vector)))
        return vector


class FlowerPrivacyAggregator:
    """Federated aggregation of privacy metrics using Flower helper utilities."""

    def __init__(self, cfg: PrivacySummaryConfig):
        self.cfg = cfg
        self.weighted_average = _lazy_import_flower()
        self.enabled = self.weighted_average is not None and cfg.federated_enabled
        self.current_threshold = cfg.risk_threshold

    def aggregate(self, metrics_batch: List[Dict[str, float]]) -> float:
        if not self.enabled or not metrics_batch:
            return self.current_threshold

        weights_results = []
        for metrics in metrics_batch:
            sensitive_ratio = float(metrics.get("pii_density", 0.0))
            removal_ratio = float(metrics.get("removal_ratio", 0.0))
            avg_risk = float(metrics.get("average_risk", 0.0))
            weight = float(metrics.get("weight", self.cfg.federated_weight))
            vector = np.array([sensitive_ratio, removal_ratio, avg_risk], dtype=np.float64)
            weights_results.append(([vector], weight))

        try:
            aggregated = self.weighted_average(weights_results)
            aggregated_vector = aggregated[0] if aggregated else np.zeros(3)
            new_threshold = float(
                np.clip(
                    aggregated_vector[0],
                    self.cfg.federated_min_threshold,
                    self.cfg.federated_max_threshold,
                )
            )
            self.current_threshold = new_threshold
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Flower aggregation failed: %s", exc)

        return self.current_threshold


class PrivacyAwareSummaryPostprocessor(BaseNodePostprocessor):
    """Main postprocessor integrating Presidio, Eraser4RAG, and TenSEAL."""

    def __init__(self, cfg: PrivacySummaryConfig):
        super().__init__()
        self.cfg = cfg
        self.presidio = PresidioSanitizer(cfg) if cfg.enable else None
        self.eraser = Eraser4RAGSanitizer(cfg) if cfg.enable else None
        self.encryptor = TenSEALEncryptor(cfg) if cfg.enable else None
        self.aggregator = FlowerPrivacyAggregator(cfg) if cfg.enable else None
        self._metrics_buffer: List[Dict[str, float]] = []

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not self.cfg.enable or not nodes:
            return nodes

        processed_nodes: List[NodeWithScore] = []
        query_text = query_bundle.query_str if query_bundle else None

        for node_with_score in nodes:
            node = node_with_score.node
            original_text = node.get_content()

            analysis = (
                self.presidio.analyze(original_text)
                if self.presidio is not None
                else {"results": [], "normalized": []}
            )

            sanitized_text = original_text
            eraser_meta = {
                "removed_count": 0,
                "kept_count": 1,
                "average_risk": 0.0,
                "max_risk": 0.0,
                "removed_sentences": [],
            }
            if self.eraser is not None:
                sanitized_text, eraser_meta = self.eraser.sanitize(
                    original_text, query_text, analysis.get("normalized", [])
                )

            anonymized_text = sanitized_text
            if self.presidio is not None:
                anonymized_text, _ = self.presidio.anonymize(sanitized_text, analysis)

            encrypted_payload = (
                self.encryptor.encrypt(anonymized_text)
                if self.encryptor is not None
                else None
            )

            privacy_metadata = self._build_privacy_metadata(
                original_text=original_text,
                sanitized_text=sanitized_text,
                anonymized_text=anonymized_text,
                analysis=analysis,
                eraser_meta=eraser_meta,
                encrypted_payload=encrypted_payload,
            )

            if self.cfg.log_stats:
                LOGGER.debug(
                    "Privacy-aware summary stats: %s",
                    {
                        "node_id": node.node_id,
                        "pii_count": len(analysis.get("normalized", [])),
                        "removed_count": eraser_meta.get("removed_count", 0),
                        "encryption": encrypted_payload is not None,
                    },
                )

            enriched_node = self._clone_node_with_text(node, anonymized_text)
            enriched_node.metadata = dict(enriched_node.metadata or {})
            enriched_node.metadata["privacy_summary"] = privacy_metadata

            patched_node_with_score = deepcopy(node_with_score)
            patched_node_with_score.node = enriched_node
            processed_nodes.append(patched_node_with_score)

            self._metrics_buffer.append(
                {
                    "pii_density": privacy_metadata["pii_density"],
                    "removal_ratio": privacy_metadata["removal_ratio"],
                    "average_risk": privacy_metadata["eraser"]["average_risk"],
                    "weight": float(len(anonymized_text)),
                }
            )

        self._maybe_update_threshold()
        return processed_nodes

    @staticmethod
    def _clone_node_with_text(node: Node, new_text: str) -> Node:
        if isinstance(node, TextNode):
            clone = deepcopy(node)
            clone.text = new_text
            return clone
        clone = deepcopy(node)
        if hasattr(clone, "text"):
            clone.text = new_text
        return clone

    def _build_privacy_metadata(
        self,
        original_text: str,
        sanitized_text: str,
        anonymized_text: str,
        analysis: Dict[str, Any],
        eraser_meta: Dict[str, Any],
        encrypted_payload: Optional[str],
    ) -> Dict[str, Any]:
        pii_items = analysis.get("normalized", [])
        pii_chars = sum(item["end"] - item["start"] for item in pii_items)
        pii_density = (
            pii_chars / max(1, len(original_text)) if original_text else 0.0
        )
        removal_ratio = (
            eraser_meta.get("removed_count", 0)
            / max(1, eraser_meta.get("removed_count", 0) + eraser_meta.get("kept_count", 0))
        )

        return {
            "pii_entities": [
                {
                    "entity_type": item["entity_type"],
                    "score": item.get("score"),
                }
                for item in pii_items
            ],
            "pii_density": float(pii_density),
            "removal_ratio": float(removal_ratio),
            "original_characters": len(original_text),
            "sanitized_characters": len(sanitized_text),
            "anonymized_characters": len(anonymized_text),
            "eraser": eraser_meta,
            "encryption": {
                "enabled": encrypted_payload is not None,
                "payload": encrypted_payload,
            },
        }

    def _maybe_update_threshold(self) -> None:
        if self.aggregator is None or not self._metrics_buffer:
            return
        new_threshold = self.aggregator.aggregate(self._metrics_buffer)
        self._metrics_buffer = []
        if self.eraser is not None:
            self.eraser.update_threshold(new_threshold)
            if self.cfg.log_stats:
                LOGGER.debug("Updated eraser risk threshold via Flower: %s", new_threshold)


class PrivacyAwareResponsePostprocessor:
    """Postprocessor for LLM response text after generation."""
    
    def __init__(self, cfg: PrivacySummaryConfig):
        self.cfg = cfg
        self.presidio = PresidioSanitizer(cfg) if cfg.enable else None
        self.eraser = Eraser4RAGSanitizer(cfg) if cfg.enable else None
        self.encryptor = TenSEALEncryptor(cfg) if cfg.enable else None
        self.aggregator = FlowerPrivacyAggregator(cfg) if cfg.enable else None
        self._metrics_buffer: List[Dict[str, float]] = []
    
    def postprocess_response(
        self, 
        response: Any, 
        query_text: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Postprocess LLM response to protect privacy.
        
        Args:
            response: LLM Response object
            query_text: Original query text (for context-aware sanitization)
        
        Returns:
            Tuple of (processed_response, privacy_metadata)
        """
        if not self.cfg.enable:
            return response, {}
        
        original_text = response.response if hasattr(response, 'response') else str(response)
        
        # Step 1: Presidio Analysis
        analysis = (
            self.presidio.analyze(original_text)
            if self.presidio is not None
            else {"results": [], "normalized": []}
        )
        
        # Step 2: Eraser4RAG Sanitization
        sanitized_text = original_text
        eraser_meta = {
            "removed_count": 0,
            "kept_count": 1,
            "average_risk": 0.0,
            "max_risk": 0.0,
            "removed_sentences": [],
        }
        if self.eraser is not None:
            sanitized_text, eraser_meta = self.eraser.sanitize(
                original_text, query_text, analysis.get("normalized", [])
            )
        
        # Step 3: Presidio Anonymization
        anonymized_text = sanitized_text
        if self.presidio is not None:
            anonymized_text, _ = self.presidio.anonymize(sanitized_text, analysis)
        
        # Step 4: TenSEAL Encryption (optional)
        encrypted_payload = (
            self.encryptor.encrypt(anonymized_text)
            if self.encryptor is not None
            else None
        )
        
        # Step 5: Build Privacy Metadata
        privacy_metadata = self._build_privacy_metadata(
            original_text=original_text,
            sanitized_text=sanitized_text,
            anonymized_text=anonymized_text,
            analysis=analysis,
            eraser_meta=eraser_meta,
            encrypted_payload=encrypted_payload,
        )
        
        # Step 6: Update Response
        processed_response = deepcopy(response)
        # Update response text
        if hasattr(processed_response, 'response'):
            processed_response.response = anonymized_text
        elif hasattr(processed_response, 'message') and hasattr(processed_response.message, 'content'):
            # For ChatResponse objects
            processed_response.message.content = anonymized_text
        
        # Add privacy metadata to response
        if not hasattr(processed_response, 'metadata') or processed_response.metadata is None:
            processed_response.metadata = {}
        if isinstance(processed_response.metadata, dict):
            processed_response.metadata["privacy_summary"] = privacy_metadata
        else:
            # If metadata is not a dict, create new dict
            processed_response.metadata = {"privacy_summary": privacy_metadata}
        
        # Log stats
        if self.cfg.log_stats:
            LOGGER.debug(
                "Privacy-aware response postprocessing stats: %s",
                {
                    "pii_count": len(analysis.get("normalized", [])),
                    "removed_count": eraser_meta.get("removed_count", 0),
                    "encryption": encrypted_payload is not None,
                },
            )
        
        # Update metrics buffer for Flower aggregation
        self._metrics_buffer.append(
            {
                "pii_density": privacy_metadata["pii_density"],
                "removal_ratio": privacy_metadata["removal_ratio"],
                "average_risk": privacy_metadata["eraser"]["average_risk"],
                "weight": float(len(anonymized_text)),
            }
        )
        
        self._maybe_update_threshold()
        
        return processed_response, privacy_metadata
    
    def _build_privacy_metadata(
        self,
        original_text: str,
        sanitized_text: str,
        anonymized_text: str,
        analysis: Dict[str, Any],
        eraser_meta: Dict[str, Any],
        encrypted_payload: Optional[str],
    ) -> Dict[str, Any]:
        """Build privacy metadata (reuse from PrivacyAwareSummaryPostprocessor)."""
        pii_items = analysis.get("normalized", [])
        pii_chars = sum(item["end"] - item["start"] for item in pii_items)
        pii_density = (
            pii_chars / max(1, len(original_text)) if original_text else 0.0
        )
        removal_ratio = (
            eraser_meta.get("removed_count", 0)
            / max(1, eraser_meta.get("removed_count", 0) + eraser_meta.get("kept_count", 0))
        )

        return {
            "pii_entities": [
                {
                    "entity_type": item["entity_type"],
                    "score": item.get("score"),
                }
                for item in pii_items
            ],
            "pii_density": float(pii_density),
            "removal_ratio": float(removal_ratio),
            "original_characters": len(original_text),
            "sanitized_characters": len(sanitized_text),
            "anonymized_characters": len(anonymized_text),
            "eraser": eraser_meta,
            "encryption": {
                "enabled": encrypted_payload is not None,
                "payload": encrypted_payload,
            },
        }
    
    def _maybe_update_threshold(self) -> None:
        """Update risk threshold via Flower aggregation."""
        if self.aggregator is None or not self._metrics_buffer:
            return
        new_threshold = self.aggregator.aggregate(self._metrics_buffer)
        self._metrics_buffer = []
        if self.eraser is not None:
            self.eraser.update_threshold(new_threshold)
            if self.cfg.log_stats:
                LOGGER.debug("Updated eraser risk threshold via Flower: %s", new_threshold)


def get_privacy_postprocessors(cfg: Any) -> List[BaseNodePostprocessor]:
    """Factory for privacy-aware postprocessors given the global config.
    
    NOTE: This function now returns empty list as privacy is applied to response, not nodes.
    Use get_privacy_response_postprocessor() instead.
    """
    # Privacy is now applied to response, not nodes
    return []


def get_privacy_response_postprocessor(cfg: Any) -> Optional[PrivacyAwareResponsePostprocessor]:
    """Factory for privacy-aware response postprocessor."""
    privacy_cfg_dict = getattr(cfg, "privacy", None)
    if isinstance(privacy_cfg_dict, dict):
        privacy_cfg = PrivacySummaryConfig.from_dict(privacy_cfg_dict)
    else:
        privacy_cfg = PrivacySummaryConfig()
    if not privacy_cfg.enable:
        return None
    return PrivacyAwareResponsePostprocessor(privacy_cfg)


def apply_privacy_to_response(
    response: Any, 
    query_text: Optional[str], 
    cfg: Any
) -> Tuple[Any, Dict[str, Any]]:
    """
    Helper function to apply privacy protection to LLM response.
    
    Args:
        response: LLM Response object
        query_text: Original query text
        cfg: Config object
    
    Returns:
        Tuple of (processed_response, privacy_metadata)
    """
    postprocessor = get_privacy_response_postprocessor(cfg)
    if postprocessor is None:
        return response, {}
    return postprocessor.postprocess_response(response, query_text)
