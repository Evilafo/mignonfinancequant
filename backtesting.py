class Backtester:
    def __init__(self, initial_portfolio, strategy):
        self.portfolio = initial_portfolio
        self.strategy = strategy
    
    def run_backtest(self, data):
        """
        Exécute un backtest sur des données historiques.
        """
        for date, prices in data.iterrows():
            self.strategy.execute(self.portfolio, prices)