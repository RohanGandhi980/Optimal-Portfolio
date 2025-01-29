import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization

TRADING_DAYS = 252
NUM_PORTFOLIOS = 1000

# Stocks to analyze
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'DB', 'AMZN', 'GOOGL']

# Date range
start_date = "2010-01-01"
end_date = "2024-01-01"


def download_data():
    """Download historical stock data from Yahoo Finance."""
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)["Close"]
    return pd.DataFrame(stock_data)


def calculate_return(data):
    """Compute log daily returns."""
    log_return = np.log(data / data.shift(1))
    return log_return.dropna()


def generate_portfolios(returns):
    """Generate random portfolios and compute their statistics."""
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def optimal_portfolio(weights, returns):
    """Compute expected return, volatility, and Sharpe ratio for given weights."""
    portfolio_return = np.sum(returns.mean() * weights) * TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * TRADING_DAYS, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])


def min_sharpe(weights, returns):
    """Objective function: Negative Sharpe ratio for minimization."""
    return -optimal_portfolio(weights, returns)[2]


def optimize_portfolio(returns):
    """Find the optimal portfolio with the highest Sharpe ratio."""
    num_stocks = len(stocks)

    # Initial guess (equal weights)
    x0 = np.ones(num_stocks) / num_stocks

    # Constraints: Weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Bounds: Weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_stocks))

    # Optimize the Sharpe ratio
    result = optimization.minimize(
        fun=min_sharpe,
        x0=x0,
        args=(returns,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result


def display_optimal(optimum, returns):
    """Print optimal portfolio weights and statistics."""
    print("Optimal Portfolio Weights:", optimum['x'].round(3))
    print("Expected Return, Volatility, Sharpe Ratio:", optimal_portfolio(optimum['x'], returns))


def graph_optimal_portfolio(opt, returns, port_rets, port_vols):
    """Plot portfolio risk-return space with optimal portfolio."""
    plt.figure(figsize=(12, 6))
    plt.scatter(port_vols, port_rets, c=port_rets / port_vols, marker='o', cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.grid(True)

    # Plot the optimal portfolio
    opt_return, opt_volatility, _ = optimal_portfolio(opt['x'], returns)
    plt.scatter(opt_volatility, opt_return, marker='*', color='red', s=300, label="Optimal Portfolio")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataset = download_data()
    log_daily_returns = calculate_return(dataset)

    pweights, means, risks = generate_portfolios(log_daily_returns)
    optimal = optimize_portfolio(log_daily_returns)

    display_optimal(optimal, log_daily_returns)
    graph_optimal_portfolio(optimal, log_daily_returns, means, risks)
