import numpy as np

def monte_carlo_option_pricing(S0, K, T, r, sigma, option_type='call', num_simulations=10000, N=252):
    """
    Évalue le prix d'une option européenne par simulation de Monte Carlo.
    
    Paramètres:
    - S0: Prix actuel du sous-jacent
    - K: Prix d'exercice
    - T: Temps jusqu'à l'échéance (en années)
    - r: Taux sans risque
    - sigma: Volatilité
    - option_type: 'call' ou 'put'
    - num_simulations: Nombre de simulations
    - N: Nombre de pas de temps
    
    Retourne:
    - Prix estimé de l'option
    """
    dt = T / N
    paths = np.zeros((num_simulations, N + 1))
    paths[:, 0] = S0
    
    for i in range(1, N + 1):
        Z = np.random.normal(size=num_simulations)
        paths[:, i] = paths[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - paths[:, -1], 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price


def asian_option_pricing(S0, K, T, r, sigma, option_type='call', num_simulations=10000, N=252):
    """
    Évalue le prix d'une option asiatique par simulation de Monte Carlo.
    
    Paramètres:
    - S0: Prix actuel du sous-jacent
    - K: Prix d'exercice
    - T: Temps jusqu'à l'échéance (en années)
    - r: Taux sans risque
    - sigma: Volatilité
    - option_type: 'call' ou 'put'
    - num_simulations: Nombre de simulations
    - N: Nombre de pas de temps
    
    Retourne:
    - Prix estimé de l'option asiatique
    """
    dt = T / N
    paths = np.zeros((num_simulations, N + 1))
    paths[:, 0] = S0
    
    for i in range(1, N + 1):
        Z = np.random.normal(size=num_simulations)
        paths[:, i] = paths[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    average_prices = np.mean(paths, axis=1)
    
    if option_type == 'call':
        payoffs = np.maximum(average_prices - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - average_prices, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def lookback_option_pricing(S0, T, r, sigma, option_type='call', num_simulations=10000, N=252):
    """
    Évalue le prix d'une option lookback par simulation de Monte Carlo.
    
    Paramètres:
    - S0: Prix actuel du sous-jacent
    - T: Temps jusqu'à l'échéance (en années)
    - r: Taux sans risque
    - sigma: Volatilité
    - option_type: 'call' ou 'put'
    - num_simulations: Nombre de simulations
    - N: Nombre de pas de temps
    
    Retourne:
    - Prix estimé de l'option lookback
    """
    dt = T / N
    paths = np.zeros((num_simulations, N + 1))
    paths[:, 0] = S0
    
    for i in range(1, N + 1):
        Z = np.random.normal(size=num_simulations)
        paths[:, i] = paths[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - np.min(paths, axis=1), 0)
    elif option_type == 'put':
        payoffs = np.maximum(np.max(paths, axis=1) - paths[:, -1], 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price


def barrier_option_pricing(S0, K, T, r, sigma, barrier, barrier_type='up-and-out', option_type='call', num_simulations=10000, N=252):
    """
    Évalue le prix d'une option barrière par simulation de Monte Carlo.
    
    Paramètres:
    - S0: Prix actuel du sous-jacent
    - K: Prix d'exercice
    - T: Temps jusqu'à l'échéance (en années)
    - r: Taux sans risque
    - sigma: Volatilité
    - barrier: Niveau de la barrière
    - barrier_type: Type de barrière ('up-and-out', 'down-and-out', 'up-and-in', 'down-and-in')
    - option_type: 'call' ou 'put'
    - num_simulations: Nombre de simulations
    - N: Nombre de pas de temps
    
    Retourne:
    - Prix estimé de l'option barrière
    """
    dt = T / N
    paths = np.zeros((num_simulations, N + 1))
    paths[:, 0] = S0
    
    for i in range(1, N + 1):
        Z = np.random.normal(size=num_simulations)
        paths[:, i] = paths[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    if barrier_type == 'up-and-out':
        valid_paths = np.max(paths, axis=1) < barrier
    elif barrier_type == 'down-and-out':
        valid_paths = np.min(paths, axis=1) > barrier
    elif barrier_type == 'up-and-in':
        valid_paths = np.max(paths, axis=1) >= barrier
    elif barrier_type == 'down-and-in':
        valid_paths = np.min(paths, axis=1) <= barrier
    else:
        raise ValueError("barrier_type invalide")
    
    final_prices = paths[:, -1]
    if option_type == 'call':
        payoffs = np.maximum(final_prices - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - final_prices, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    discounted_payoffs = np.exp(-r * T) * payoffs
    option_price = np.mean(discounted_payoffs[valid_paths])
    return option_price