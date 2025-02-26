# Optimal-Portfolio

This project is a portfolio optimization system that applies modern portfolio theory (MPT) to maximize risk-adjusted returns using historical stock data. It calculates log daily returns for risk-return analysis.

The project simulates 1,000 random portfolios, assigning different weight distributions to stocks. It computes each portfolio’s expected return, volatility (risk), and Sharpe ratio—a measure of risk-adjusted performance. Using SciPy’s Sequential Least Squares Programming (SLSQP) optimizer, it finds the optimal portfolio allocation that maximizes the Sharpe ratio

The script visualizes the risk-return tradeoff by plotting portfolios in a risk-return space, highlighting the optimal portfolio. The project is a practical implementation of efficient frontier concepts, aiding in intelligent investment decisions.
