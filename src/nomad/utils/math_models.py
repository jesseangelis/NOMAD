import numpy as np
from numba import njit
from scipy import stats

class LogisticModel:
    """Core log-logistic model for fitting and prediction."""
    
    @staticmethod
    @njit(nogil=True, fastmath=True)
    def core(x_vec: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """4-parameter log-logistic model: f(x) = c + (d-c) / (1 + 10^(b(a-x)))."""
        exponent = b * (a - x_vec)
        # Clip exponent to avoid overflow in 10**x
        exponent = np.maximum(np.minimum(exponent, 20.0), -20.0)
        return c + (d - c) / (1.0 + 10.0**exponent)

    @staticmethod
    @njit(nogil=True, fastmath=True)
    def jacobian(x_vec: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """Partial derivatives wrt (a, b, c, d)."""
        l10 = np.log(10.0)
        # 10^(b(a-x))
        exponent = b * (a - x_vec)
        exp_val = 10.0**np.maximum(np.minimum(exponent, 20.0), -20.0)
        # Denom = (1 + 10^(b(a-x)))
        denom = 1.0 + exp_val
        
        # d/da = (c-d) * l10 * b * exp_val / denom^2
        da = (c - d) * l10 * b * exp_val / (denom**2)
        # d/db = (c-d) * l10 * (a-x) * exp_val / denom^2
        db = (c - d) * l10 * (a - x_vec) * exp_val / (denom**2)
        # d/dc = 1 - 1/denom = exp_val / denom
        dc = exp_val / denom
        # d/dd = 1/denom
        dd = 1.0 / denom
        
        return np.stack((da, db, dc, dd), axis=-1)

def get_sam_relevance_score(rss_null: float, rss_alt: float, n_eff: float, lfc: float,
                          s0_noise: float = 0.01, 
                          s0_mag: float = 0.1) -> tuple:
    """Calculates the Jointly-Punished SAM Relevance Score.
    
    This framework combines NOMAD's Ambiguity Penalty (n_eff) with 
    CurveCurator's Magnitude Penalty (s0_mag).
    
    n_eff: Kish's Effective Sample Size (accounts for deconvolution uncertainty).
    lfc: Log2 Fold Change (min vs. max dose).
    s0_noise: Stabilizing factor for the F-statistic denominator (noise floor).
    s0_mag: SAM fudge factor to penalize low-magnitude biological noise.
    """
    if n_eff <= 4.0:
        return 0.0, 1.0
    
    # 1. Ambiguity-Aware F-statistic (NOMAD core) - df1 corrected to 3.0
    df2 = np.maximum(1.0, n_eff - 4.0)
    f_nomad = ((rss_null - rss_alt) / 3.0 + 1e-12) / ((rss_alt + s0_noise) / df2 + 1e-12)
    f_nomad = np.maximum(0.0, f_nomad)
    
    # 2. Magnitude-Aware SAM Correction (CurveCurator core)
    # F_adj = 1 / ((1/sqrt(F)) + (s0_mag / |lfc|))^2
    lfc_abs = np.abs(lfc)
    f_joint = 1.0 / ((1.0 / (np.sqrt(f_nomad) + 1e-12)) + (s0_mag / (lfc_abs + 1e-12)))**2
    
    # 3. Final P-value and Unsigned Relevance Score - df1 corrected to 3.0
    p_val = stats.f.sf(f_joint, 3.0, df2)
    rs = -np.log10(p_val + 1e-300)
    
    return float(rs), float(p_val)
