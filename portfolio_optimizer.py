import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
    
    # Flatten columns if multi-index (which happens with multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)

    returns = data.pct_change().dropna()
    return returns

#portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns) * 252  # Annualized return
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
    sharpe = ret / vol
    return ret, vol, sharpe

#optimize for max sharpe ratio
def negative_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    init_guess = num_assets * [1. / num_assets]
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    result = minimize(negative_sharpe, init_guess, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

#simulate random portfolios (efficient frontier)
def simulate_portfolios(mean_returns, cov_matrix, num_portfolios=5000):
    results = {'returns': [], 'volatility': [], 'sharpe': [], 'weights': []}
    num_assets = len(mean_returns)

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
        results['returns'].append(ret)
        results['volatility'].append(vol)
        results['sharpe'].append(sharpe)
        results['weights'].append(weights)

    return results

#Monte Carlo Functions
def simulate_price_paths(S0, mu, sigma, T=1, steps=252, n_paths=10000):
    dt = T / steps
    prices = np.zeros((steps + 1, n_paths))
    prices[0] = S0
    for t in range(1, steps + 1):
        Z = np.random.standard_normal(n_paths)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return prices

def monte_carlo_portfolio_distribution(initial_prices, mean_returns, cov_matrix, weights, T=1, steps=252, n_paths=10000):
    n_assets = len(initial_prices)
    final_prices = []

    for i in range(n_assets):
        sim_paths = simulate_price_paths(
            S0=initial_prices[i],
            mu=mean_returns[i],
            sigma=np.sqrt(cov_matrix.iloc[i, i]),
            T=T,
            steps=steps,
            n_paths=n_paths
        )
        final_prices.append(sim_paths[-1])  # final prices only

    final_prices = np.array(final_prices)  # shape: [n_assets, n_paths]

    # Portfolio value = dot product of weights × final simulated prices
    portfolio_values = np.dot(weights, final_prices)

    # Scale the portfolio values to match starting capital (optional but realistic)
    initial_portfolio_value = np.dot(initial_prices, weights)
    portfolio_values = portfolio_values / np.dot(initial_prices, weights) * 100000  # start at $100k baseline

    return portfolio_values

def plot_both(ef_results, optimal_weights, mean_returns, cov_matrix, portfolio_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Efficient Frontier
    sc = ax1.scatter(ef_results['volatility'], ef_results['returns'],
                     c=ef_results['sharpe'], cmap='viridis', alpha=0.7)
    opt_ret, opt_vol, _ = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    ax1.scatter(opt_vol, opt_ret, c='red', s=100, marker='*', label='Optimal Portfolio')
    ax1.set_title('Efficient Frontier')
    ax1.set_xlabel('Annualized Volatility')
    ax1.set_ylabel('Annualized Return')
    ax1.legend()
    fig.colorbar(sc, ax=ax1, label='Sharpe Ratio')

    # Monte Carlo Distribution
    ax2.hist(portfolio_values, bins=50, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(portfolio_values), color='red', linestyle='--', label='Mean Value')
    ax2.set_title('Monte Carlo Future Portfolio Value')
    ax2.set_xlabel('Portfolio Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    returns = load_data(tickers, '2020-01-01', '2024-12-31')

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Optimize portfolio
    opt_result = optimize_portfolio(mean_returns, cov_matrix)
    optimal_weights = opt_result.x

    # Simulate random portfolios
    sim_results = simulate_portfolios(mean_returns, cov_matrix)

    # Monte Carlo simulation of future portfolio value
    initial_prices = returns.iloc[-1].values
    sim_values = monte_carlo_portfolio_distribution(
        initial_prices=initial_prices,
        mean_returns=mean_returns.values,
        cov_matrix=cov_matrix,
        weights=optimal_weights,
        T=1,
        steps=252,
        n_paths=10000
    )

    initial_portfolio_value = np.dot(initial_prices, optimal_weights)
    sim_portfolio_values = initial_portfolio_value * (1 + sim_values)  # adjust to reflect future dollar value

    plot_both(sim_results, optimal_weights, mean_returns, cov_matrix, sim_values)

    # Print optimal weights
    print("\n✅ Optimal Portfolio Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.2%}")
    
    