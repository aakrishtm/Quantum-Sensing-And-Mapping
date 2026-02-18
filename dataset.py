"""Backward-compat re-export; use graviq.data instead."""
from graviq.data import TunnelDataset, get_dataloaders

__all__ = ["TunnelDataset", "get_dataloaders"]
