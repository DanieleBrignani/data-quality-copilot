"""Reproducible synthetic demonstration datasets."""

from dqcopilot.demodata.generator import (
    DATASETS,
    SEED,
    SeededError,
    build_customers,
    build_sales,
    build_suppliers,
    generate_all,
    load_ground_truth,
    write_demo_data,
)

__all__ = [
    "DATASETS",
    "SEED",
    "SeededError",
    "build_customers",
    "build_sales",
    "build_suppliers",
    "generate_all",
    "load_ground_truth",
    "write_demo_data",
]
