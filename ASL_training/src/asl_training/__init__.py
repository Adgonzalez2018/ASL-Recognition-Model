"""Isolated American Sign Language video classification: training and evaluation.

Layer packages:

    models      architecture adapters and the shared classification contract
    data        dataset parsing, manifests, decoding, and transforms
    training    supervised fine-tuning orchestration
    evaluation  metrics, calibration, and selective prediction
    experiments configuration composition and run records
    utils       shared helpers

Dependencies point downward. Lower layers must not import higher ones.
See docs/ARCHITECTURE.md.
"""

__version__ = "0.1.0"
