from scipy import exp, clip

def safe_exp(x):
    """to avoid inf and NaN values"""
    return exp(clip(x, -500, 500))