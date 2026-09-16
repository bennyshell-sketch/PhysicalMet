"""Continual-learning layer: ingestion, model registry, gated retraining."""

from .ingest import DataSource, IngestResult, SourceRegistry, ingest_file
from .registry import ModelRegistry, ModelVersion
from .autolearn import AutoLearner, CycleResult
from .feature_search import FeatureSearchResult, search_features
from .retrain import RetrainOutcome, retrain_and_evaluate

__all__ = [
    "DataSource",
    "IngestResult",
    "SourceRegistry",
    "ingest_file",
    "ModelRegistry",
    "ModelVersion",
    "AutoLearner",
    "CycleResult",
    "FeatureSearchResult",
    "search_features",
    "RetrainOutcome",
    "retrain_and_evaluate",
]
