# Compass

A strategy research engine and a daily brief for a portfolio you trade by hand in Robinhood.

- **Research:** 14 strategies backtested on daily prices since 2000. They range from index buy & hold through leverage, 200-day trend filters and momentum, and the simulation includes leverage costs, ETF fees and slippage. Simulated leveraged ETFs track the real SSO/UPRO/QLD/TQQQ within about 0.3%/yr.
- **Daily brief:** applies the strategy you chose to the latest close and says, in Robinhood tickers, whether to trade today and what to hold.

Compass never connects to your brokerage. Its output is the mechanical result of rules you chose, not personalized financial advice.

## Usage

```bash
pip install -r requirements.txt
python -m compass.report            # writes report/results.json, prints a summary
python -m compass.brief trend3x --value 25000   # today's brief for one strategy
python -m pytest -q tests
```

Rebuild the interactive report (`report/compass-report.html`) by compiling `report/body.html` with `report/results.json` as its data.

Strategy keys: `spy 6040 qqq scv spy15 upro tqqq hfea trend trend2x trend3x trendq2x gem sectors` (defined in `compass/strategies.py`).

Data comes from Yahoo Finance daily adjusted closes and is cached in `data/` for 12 hours.
