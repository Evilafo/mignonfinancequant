import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calcule le prix d'une option européenne avec le modèle Black-Scholes.
    
    Paramètres:
    - S: Prix actuel du sous-jacent
    - K: Prix d'exercice
    - T: Temps jusqu'à l'échéance (en années)
    - r: Taux sans risque
    - sigma: Volatilité implicite
    - option_type: 'call' ou 'put'
    
    Retourne:
    - Prix de l'option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    return price

def greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calcule les sensibilités (Greeks) d'une option.
    
    Retourne un dictionnaire contenant Delta, Gamma, Vega, Theta, Rho.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
             r * K * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'call' else (
                 -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else (
        -K * T * np.exp(-r * T) * norm.cdf(-d2))
    
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }

def american_option_binomial_tree(S0, K, T, r, sigma, option_type='call', N=100):
    """
    Évalue le prix d'une option américaine en utilisant un arbre binomial.
    
    Paramètres:
    - S0: Prix actuel du sous-jacent
    - K: Prix d'exercice
    - T: Temps jusqu'à l'échéance (en années)
    - r: Taux sans risque
    - sigma: Volatilité
    - option_type: 'call' ou 'put'
    - N: Nombre de pas dans l'arbre
    
    Retourne:
    - Prix estimé de l'option américaine
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialisation des prix terminaux
    stock_prices = S0 * (u ** np.arange(N + 1)) * (d ** np.arange(N, -1, -1))
    
    if option_type == 'call':
        option_values = np.maximum(stock_prices - K, 0)
    elif option_type == 'put':
        option_values = np.maximum(K - stock_prices, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    # Remontée dans l'arbre
    for i in range(N - 1, -1, -1):
        stock_prices = S0 * (u ** np.arange(i + 1)) * (d ** np.arange(i, -1, -1))
        continuation_values = np.exp(-r * dt) * (p * option_values[1:i+2] + (1 - p) * option_values[0:i+1])
        
        if option_type == 'call':
            option_values = np.maximum(continuation_values, stock_prices - K)
        elif option_type == 'put':
            option_values = np.maximum(continuation_values, K - stock_prices)
    
    return option_values[0]