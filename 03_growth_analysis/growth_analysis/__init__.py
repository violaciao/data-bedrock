"""Growth analysis library: funnel and acquisition analytics."""

from .acquisition import acquisition_metrics, channel_comparison
from .funnel import FunnelAnalysis

__all__ = ["FunnelAnalysis", "acquisition_metrics", "channel_comparison"]
