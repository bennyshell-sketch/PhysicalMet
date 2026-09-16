"""Continual-learning layer: ingestion, model registry, gated retraining."""

from .ingest import DataSource, IngestResult, SourceRegistry, ingest_file
from .registry import ModelRegistry, ModelVersion
from .feature_search import FeatureSearchResult, search_features
from .retrain import RetrainOutcome, retrain_and_evaluate

__all__ = [
    "DataSource",
    "IngestResult",
    "SourceRegistry",
    "ingest_file",
    "ModelRegistry",
    "ModelVersion",
    "FeatureSearchResult",
    "search_features",
    "RetrainOutcome",
    "retrain_and_evaluate",
]
