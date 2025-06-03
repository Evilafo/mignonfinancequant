import numpy as np

def geometric_brownian_motion(S0, mu, sigma, T, N, num_simulations=1):
    """
    Simule un mouvement brownien géométrique.
    
    Paramètres:
    - S0: Prix initial
    - mu: Taux de dérive (rendement attendu)
    - sigma: Volatilité
    - T: Horizon temporel (en années)
    - N: Nombre de pas de temps
    - num_simulations: Nombre de simulations
    
    Retourne:
    - Tableau des trajectoires simulées (shape: num_simulations x N+1)
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    paths = np.zeros((num_simulations, N + 1))
    paths[:, 0] = S0
    
    for i in range(1, N + 1):
        Z = np.random.normal(size=num_simulations)
        paths[:, i] = paths[:, i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return t, paths


def ornstein_uhlenbeck_process(X0, theta, mu, sigma, T, N, num_simulations=1):
    """
    Simule un processus d'Ornstein-Uhlenbeck (mean-reverting process).
    
    Paramètres:
    - X0: Valeur initiale
    - theta: Vitesse de retour à la moyenne
    - mu: Moyenne de long terme
    - sigma: Volatilité
    - T: Horizon temporel (en années)
    - N: Nombre de pas de temps
    - num_simulations: Nombre de simulations
    
    Retourne:
    - Tableau des trajectoires simulées (shape: num_simulations x N+1)
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    paths = np.zeros((num_simulations, N + 1))
    paths[:, 0] = X0
    
    for i in range(1, N + 1):
        Z = np.random.normal(size=num_simulations)
        paths[:, i] = paths[:, i - 1] + theta * (mu - paths[:, i - 1]) * dt + sigma * np.sqrt(dt) * Z
    
    return t, paths