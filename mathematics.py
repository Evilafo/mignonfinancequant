import numpy as np

def historical_volatility(returns, annualized=True):
    """
    Calcule la volatilité historique.
    """
    volatility = np.std(returns)
    return volatility * np.sqrt(252) if annualized else volatility

def adjusted_return(returns, risk_free_rate=0.0):
    """
    Calcule le rendement ajusté au risque.
    """
    return np.mean(returns) - risk_free_rate