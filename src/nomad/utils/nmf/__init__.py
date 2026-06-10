"""GPU-accelerated Joint Non-negative Matrix Factorization solver suite.

Provides parallel multi-GPU/multi-threaded decomposition of biological mass spectrometry
evidence graphs with structural manifold regularizers and replicate variance penalties.
"""

from nomad.utils.nmf.engine import NMFFit

__all__ = ["NMFFit"]
