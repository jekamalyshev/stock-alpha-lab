# Stock Alpha Lab

Research toolkit for building equity trading strategies with positive expectancy on 5-minute OHLCV data (Sberbank / MOEX).

## Structure

```
stock-alpha-lab/
├── features/
│   └── candle_features.py   # Feature engineering pipeline
└── README.md
```

## Feature Engineering

`build_sber_5m_candle_features` accepts a DataFrame with columns:

```
TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL
```

Returns the same DataFrame enriched with **~130+ features** across 5 groups:

| Group | Count | Examples |
|---|---|---|
| Candle geometry | ~16 | `body_to_range`, `upper_wick`, `close_pos_in_range` |
| Returns / momentum | ~10 | `ret_1`, `ret_12`, `gap_from_prev_close` |
| Rolling context | ~11 × windows | `range_zscore_6`, `vol_ratio_6`, `close_vs_ma_12` |
| Volume interaction | ~4 | `signed_vol`, `ret1_x_volratio_6` |
| TA indicators (ta lib) | ~80+ | `rsi`, `macd`, `atr`, `bb_pct`, `vwap`, `obv`, `mfi` |
| Target (shift=4) | 1 | `target_ret_4` — **future data, label only** |

### Quick start

```python
pip install pandas numpy ta
```

```python
import pandas as pd
from features.candle_features import build_sber_5m_candle_features

df = pd.read_csv(
    "SBER_5min.csv",
    names=["TICKER", "PER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL"],
)

df_feat = build_sber_5m_candle_features(
    df,
    windows=(3, 6, 12),
    ta_window=14,
    shift=4,          # target = close pct_change(4) shifted -4 bars = +20 min
    verbose=True,
)

print(df_feat.shape)  # (rows, ~140+)
```

### Sample output

```
============================================================
  Feature Engineering Report
============================================================
  Columns before  : 9
  New features    : ~130
  Columns after   : ~139
  ├── incl. target_ret_4 (future data - DO NOT use as feature!)
============================================================
```

## Key design decisions

- **No look-ahead bias** — all features use only current and past bar data
- **shift=4** (default) → target is close return 4 bars (20 min) ahead
- `target_ret_N` column is labeled with `N` so you always know the horizon
- Flat zero candle ranges guarded with `eps` to avoid division by zero
- `copy=True` by default — input DataFrame is never mutated

## Principles

> A strategy has value only when it shows positive expectancy after accounting for all costs:
> `Expectancy = (Win Rate × Avg Win) − (Loss Rate × Avg Loss)`

Always account for:
- Commissions + slippage
- Liquidity constraints
- Survivorship bias
- Look-ahead / data leakage
- Regime shifts

## License

MIT
