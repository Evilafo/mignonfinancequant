class Portfolio:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.positions = {}
    
    def add_position(self, asset, quantity, price):
        """
        Ajoute une position dans le portefeuille.
        """
        self.positions[asset] = self.positions.get(asset, 0) + quantity
        self.cash -= quantity * price
    
    def remove_position(self, asset, quantity, price):
        """
        Supprime une position du portefeuille.
        """
        if asset in self.positions and self.positions[asset] >= quantity:
            self.positions[asset] -= quantity
            self.cash += quantity * price
        else:
            raise ValueError("Quantité insuffisante pour supprimer la position.")
    
    def portfolio_value(self, prices):
        """
        Calcule la valeur totale du portefeuille.
        """
        value = self.cash
        for asset, quantity in self.positions.items():
            value += quantity * prices.get(asset, 0)
        return value