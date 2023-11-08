import random

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
import numpy as np
import pandas as pd

from src.tendies_maker.gather import get_price_history, get_news, get_macro_econ_data, \
    get_options_snapshot
from src.tendies_maker.thinker import append_technical_indicators, news_sentiment_analysis, append_options_metrics, \
    append_fluctuations
from src.tendies_maker.datamodel import TrainingData as td

ticker = 'SPY'
details_contract_type = 'put'
options_chain = get_options_snapshot(ticker)
options_chain = options_chain[options_chain['details_contract_type'] == details_contract_type]
options_chain, metrics = append_options_metrics(options_chain)


class OptimalContractProblem(Problem):

    def __init__(self, options_chain):
        super().__init__(n_var=1,  # Number of decision variables (1 in this case for the index)
                         n_obj=3,  # Number of objectives
                         n_constr=2,  # Number of constraints (volume and open interest)
                         xl=np.array([0]),  # Lower bound
                         xu=np.array([len(options_chain) - 1])  # Upper bound
                         )
        self.options_chain = options_chain

    def _evaluate(self, x, out, *args, **kwargs):
        # Objective 1: Maximize moneyness
        moneyness = self.options_chain.iloc[x[:, 0].astype(int)]['moneyness'].values

        # Objective 2: Minimize theta per delta
        theta_per_delta = self.options_chain.iloc[x[:, 0].astype(int)]['theta_per_delta'].values

        # Objective 3: Minimize cost
        cost = self.options_chain.iloc[x[:, 0].astype(int)]['day_open'].values

        out['F'] = np.column_stack([moneyness, -theta_per_delta, -cost])

        # Initialize 'G' if it's None
        if out['G'] is None:
            out['G'] = np.zeros((x.shape[0], 2))

        # Constraint 1: Volume should be greater than a threshold (e.g., 100)
        out['G'][:, 0] = 100 - self.options_chain.iloc[x[:, 0].astype(int)]['day_volume'].values

        # Constraint 2: Open interest should be greater than a threshold (e.g., 100)
        out['G'][:, 1] = 100 - self.options_chain.iloc[x[:, 0].astype(int)]['open_interest'].values



problem = OptimalContractProblem(options_chain)

# Initialize the algorithm
algorithm = NSGA2(pop_size=100)

# Run the optimization
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               verbose=False)


best_indices = res.X.astype(int).flatten()
best_options = options_chain.iloc[best_indices]
best_options = best_options[['day_change_percent', 'day_open', 'day_high', 'day_vwap', 'day_volume',
                             'details_contract_type', 'details_expiration_date', 'details_strike_price',
                             'greeks_delta', 'greeks_gamma', 'greeks_theta', 'greeks_vega','implied_volatility',
                             'open_interest', 'theta_per_delta', 'moneyness', 'days_to_expiration']]
breakpoint()
