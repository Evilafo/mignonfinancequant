import numpy as np

def calculate_var(returns, confidence_level=0.95):
    """
    Calcule la Value at Risk (VaR).
    
    Paramètres:
    - returns: Tableau des rendements
    - confidence_level: Niveau de confiance (par défaut 95%)
    
    Retourne:
    - VaR
    """
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_cvar(returns, confidence_level=0.95):
    """
    Calcule la Conditional Value at Risk (CVaR ou TVaR).
    """
    var = calculate_var(returns, confidence_level)
    tail_losses = returns[returns <= var]
    return np.mean(tail_losses)

def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calcule le ratio de Sharpe.
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calcule le ratio de Sortino.
    """
    excess_returns = returns - risk_free_rate
    downside_risk = np.sqrt(np.mean(np.minimum(excess_returns, 0)**2))
    return np.mean(excess_returns) / downside_risk

def beta_market(portfolio_returns, market_returns):
    """
    Calcule le bêta du portefeuille par rapport au marché.
    """
    covariance = np.cov(portfolio_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance