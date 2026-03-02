"""
pipeline — Windows Log Intelligence DL pipeline package
"""
from pipeline.model    import Autoencoder
from pipeline.features import build_features
from pipeline.detector import run_pipeline, severity_label, DEVICE
from pipeline.report   import write_report

__all__ = ["Autoencoder", "build_features", "run_pipeline",
           "severity_label", "DEVICE", "write_report"]
