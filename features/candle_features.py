"""5-minute OHLCV candle feature engineering for equity research.

Input columns: TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL

Dependencies:
    pip install pandas numpy ta

Feature count example (windows=(3,6,12), ta_window=14, n_in=3):
    Raw input columns        :   9
    New engineered features  :  ~61   (without TA) / ~130+ (with TA)
    Total after engineering  :  ~71 / ~139
    After series_to_supervised(n_in=3):
        non-target cols × (n_in+1) + 1 target
        e.g. 70 × 4 + 1 = 281 columns
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:
    import ta
    from ta.momentum import (
        RSIIndicator,
        StochasticOscillator,
        ROCIndicator,
        WilliamsRIndicator,
        AwesomeOscillatorIndicator,
    )
    from ta.trend import (
        MACD,
        CCIIndicator,
        EMAIndicator,
        SMAIndicator,
        ADXIndicator,
        AroonIndicator,
    )
    from ta.volatility import (
        BollingerBands,
        AverageTrueRange,
        KeltnerChannel,
        DonchianChannel,
        UlcerIndex,
    )
    from ta.volume import (
        OnBalanceVolumeIndicator,
        MFIIndicator,
        ChaikinMoneyFlowIndicator,
        EaseOfMovementIndicator,
        VolumeWeightedAveragePrice,
    )

    _TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-12
_REQUIRED_COLS = {"OPEN", "HIGH", "LOW", "CLOSE", "VOL"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def series_to_supervised(
    df: pd.DataFrame,
    n_in: int = 3,
    target_col: str = "target_ret_4",
    dropnan: bool = True,
) -> pd.DataFrame:
    """Transform an engineered feature DataFrame into a supervised learning frame.

    For each feature column creates ``n_in`` lagged copies::

        col(t-n_in), ..., col(t-1), col(t)

    The target column is carried through as-is (it already encodes a forward
    return via ``shift(-N)`` and must NOT be shifted again here).

    Parameters
    ----------
    df         : DataFrame with engineered features + pre-built target column.
    n_in       : Number of lag steps (lookback window). Default = 3.
    target_col : Name of the forward-return column built by
                 ``build_sber_5m_candle_features``.  Must exist in ``df``.
    dropnan    : Drop rows that contain NaN introduced by lagging.

    Returns
    -------
    pd.DataFrame
        Shape: (rows - n_in - NaN_rows,
                n_feature_cols * (n_in + 1) + 1)

    Column naming convention::

        CLOSE(t-3), CLOSE(t-2), CLOSE(t-1), CLOSE(t), target_ret_4

    Notes
    -----
    Column count formula::

        n_non_target = total_cols - 1
        output_cols  = n_non_target * (n_in + 1) + 1 (target)

    Example with n_in=3 and 70 feature cols + 1 target::

        70 * (3 + 1) + 1 = 281 columns
    """
    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found. "
            "Run build_sber_5m_candle_features first."
        )

    feature_cols = [c for c in df.columns if c != target_col]

    cols: list[pd.DataFrame] = []
    names: list[str] = []

    # lagged copies: t-n_in ... t-1
    for i in range(n_in, 0, -1):
        cols.append(df[feature_cols].shift(i))
        names += [f"{c}(t-{i})" for c in feature_cols]

    # current timestep: t
    cols.append(df[feature_cols])
    names += [f"{c}(t)" for c in feature_cols]

    # target carried through unchanged
    cols.append(df[[target_col]])
    names += [target_col]

    result = pd.concat(cols, axis=1)
    result.columns = names

    if dropnan:
        result = result.dropna().reset_index(drop=True)

    return result


def build_sber_5m_candle_features(
    df: pd.DataFrame,
    *,
    windows: Iterable[int] = (3, 6, 12),
    ta_window: int = 14,
    shift: int = 4,
    add_target: bool = True,
    n_in: int = 3,
    apply_supervised: bool = True,
    drop_invalid_rows: bool = False,
    copy: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build comprehensive candle + TA features for 5-minute OHLCV data.

    Pipeline stages
    ---------------
    1. Validate & cast numeric columns.
    2. Sort by datetime (DATE + TIME).
    3. Add candle geometry features.
    4. Add return / momentum features.
    5. Add rolling context features (price, range, volume).
    6. Add volume interaction features.
    7. Add TA indicators (requires ``pip install ta``).
    8. Append forward-return target (``shift(-N)`` bars, label only).
    9. Apply ``series_to_supervised`` to create lagged feature copies.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe. Expected columns:
        TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL
    windows : Iterable[int], default=(3, 6, 12)
        Rolling windows in bars. For 5-minute data:
        3 = 15 min, 6 = 30 min, 12 = 60 min.
    ta_window : int, default=14
        Base period for TA indicators (RSI, ATR, ADX, etc.).
    shift : int, default=4
        Bars ahead for the forward-return target (= 20 min for 5m bars).
    add_target : bool, default=True
        Append ``target_ret_{shift}`` column. Contains future data — label only.
    n_in : int, default=3
        Lookback window for ``series_to_supervised``. Creates ``n_in`` lagged
        copies of every feature column.
    apply_supervised : bool, default=True
        If True, call ``series_to_supervised`` after feature engineering.
    drop_invalid_rows : bool, default=False
        Drop rows where OHLCV columns are NaN after numeric casting.
    copy : bool, default=True
        Work on a deep copy of the input (recommended).
    verbose : bool, default=True
        Print feature count report with formula verification.

    Returns
    -------
    pd.DataFrame
        If ``apply_supervised=True``:
            columns = n_feature_cols * (n_in + 1) + 1 (target)
        Else:
            columns = raw_cols + engineered_features + 1 (target)

    Notes
    -----
    All features use only current and past bar data — no look-ahead bias.
    The target column is the ONLY column containing future information.
    """
    _validate_columns(df)

    out = df.copy(deep=True) if copy else df
    out = _cast_numeric(out, drop_invalid_rows)
    out = _sort_by_time(out)

    n_cols_raw = out.shape[1]

    out = _add_candle_shape(out)
    out = _add_returns(out)
    out = _add_rolling_context(out, list(windows))
    out = _add_volume_features(out)

    if _TA_AVAILABLE:
        out = _add_ta_indicators(out, ta_window=ta_window)
    else:  # pragma: no cover
        import warnings
        warnings.warn(
            "ta library not found. Install with: pip install ta",
            ImportWarning,
            stacklevel=2,
        )

    target_col = f"target_ret_{shift}"
    if add_target and shift > 0:
        out[target_col] = out["CLOSE"].pct_change(shift).shift(-shift)

    n_cols_after_engineering = out.shape[1]
    n_new_features = n_cols_after_engineering - n_cols_raw

    if apply_supervised and add_target and shift > 0:
        n_non_target = n_cols_after_engineering - 1
        expected_supervised_cols = n_non_target * (n_in + 1) + 1
        out = series_to_supervised(out, n_in=n_in, target_col=target_col)
        n_cols_supervised = out.shape[1]
    else:
        expected_supervised_cols = None
        n_cols_supervised = None

    if verbose:
        _print_feature_report(
            n_cols_raw=n_cols_raw,
            n_new_features=n_new_features,
            n_cols_after_engineering=n_cols_after_engineering,
            apply_supervised=apply_supervised,
            n_in=n_in,
            expected_supervised_cols=expected_supervised_cols,
            n_cols_supervised=n_cols_supervised,
            target_col=target_col if add_target and shift > 0 else None,
        )

    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_columns(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _cast_numeric(df: pd.DataFrame, drop_invalid: bool) -> pd.DataFrame:
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if drop_invalid:
        df = df.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]).reset_index(drop=True)
    return df


def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    if not {"DATE", "TIME"}.issubset(df.columns):
        return df
    date_str = df["DATE"].astype(str).str.strip()
    time_str = df["TIME"].astype(str).str.zfill(6).str.strip()
    df["datetime"] = pd.to_datetime(
        date_str + time_str, format="%Y%m%d%H%M%S", errors="coerce"
    )
    return df.sort_values("datetime", kind="stable").reset_index(drop=True)


def _add_candle_shape(df: pd.DataFrame) -> pd.DataFrame:
    rng = df["HIGH"] - df["LOW"]
    safe_rng = rng.where(rng.abs() > _EPS, np.nan)
    max_oc = np.maximum(df["OPEN"], df["CLOSE"])
    min_oc = np.minimum(df["OPEN"], df["CLOSE"])

    df["candle_body"]         = df["CLOSE"] - df["OPEN"]
    df["body_abs"]            = df["candle_body"].abs()
    df["candle_range"]        = rng
    df["upper_wick"]          = df["HIGH"] - max_oc
    df["lower_wick"]          = min_oc - df["LOW"]
    df["body_to_range"]       = (df["body_abs"]   / safe_rng).clip(0.0, 1.0)
    df["upper_wick_to_range"] = (df["upper_wick"] / safe_rng).clip(0.0, 1.0)
    df["lower_wick_to_range"] = (df["lower_wick"] / safe_rng).clip(0.0, 1.0)
    df["direction"]           = np.sign(df["candle_body"]).astype("float64")
    df["is_green"]            = (df["CLOSE"] > df["OPEN"]).astype("int8")
    df["is_red"]              = (df["CLOSE"] < df["OPEN"]).astype("int8")
    df["is_doji_like"]        = (df["body_to_range"] <= 0.10).astype("int8")
    df["close_pos_in_range"]  = ((df["CLOSE"] - df["LOW"]) / safe_rng).clip(0.0, 1.0)
    df["open_pos_in_range"]   = ((df["OPEN"]  - df["LOW"]) / safe_rng).clip(0.0, 1.0)
    df["range_pct_close"]     = df["candle_range"] / df["CLOSE"].replace(0.0, np.nan)
    df["range_pct_open"]      = df["candle_range"] / df["OPEN"].replace(0.0, np.nan)
    return df


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["ret_1"]               = df["CLOSE"].pct_change(1)
    df["ret_3"]               = df["CLOSE"].pct_change(3)
    df["ret_6"]               = df["CLOSE"].pct_change(6)
    df["ret_12"]              = df["CLOSE"].pct_change(12)
    df["gap_from_prev_close"] = df["OPEN"]  / df["CLOSE"].shift(1) - 1.0
    df["close_to_prev_high"]  = df["CLOSE"] / df["HIGH"].shift(1)  - 1.0
    df["close_to_prev_low"]   = df["CLOSE"] / df["LOW"].shift(1)   - 1.0
    return df


def _add_rolling_context(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"range_mean_{w}"]    = df["candle_range"].rolling(w, min_periods=w).mean()
        df[f"range_std_{w}"]     = df["candle_range"].rolling(w, min_periods=w).std()
        df[f"body_abs_mean_{w}"] = df["body_abs"].rolling(w, min_periods=w).mean()
        df[f"vol_ma_{w}"]        = df["VOL"].rolling(w, min_periods=w).mean()
        df[f"vol_std_{w}"]       = df["VOL"].rolling(w, min_periods=w).std()
        df[f"vol_ratio_{w}"]     = df["VOL"] / df[f"vol_ma_{w}"]
        df[f"close_ma_{w}"]      = df["CLOSE"].rolling(w, min_periods=w).mean()
        df[f"close_std_{w}"]     = df["CLOSE"].rolling(w, min_periods=w).std()
        df[f"close_vs_ma_{w}"]   = df["CLOSE"] / df[f"close_ma_{w}"] - 1.0
        df[f"range_zscore_{w}"]  = (
            (df["candle_range"] - df[f"range_mean_{w}"]) /
            df[f"range_std_{w}"].replace(0.0, np.nan)
        )
        df[f"close_zscore_{w}"]  = (
            (df["CLOSE"] - df[f"close_ma_{w}"]) /
            df[f"close_std_{w}"].replace(0.0, np.nan)
        )
    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_chg_1"]         = df["VOL"].pct_change(1)
    df["body_x_vol"]        = df["body_abs"] * df["VOL"]
    df["signed_vol"]        = df["direction"] * df["VOL"]
    vol_ratio_6             = df.get("vol_ratio_6", pd.Series(np.nan, index=df.index))
    df["ret1_x_volratio_6"] = df["ret_1"] * vol_ratio_6
    return df


def _add_ta_indicators(df: pd.DataFrame, ta_window: int = 14) -> pd.DataFrame:
    c, h, lo, v = df["CLOSE"], df["HIGH"], df["LOW"], df["VOL"]

    # Momentum
    df["rsi"]          = RSIIndicator(close=c, window=ta_window).rsi()
    stoch              = StochasticOscillator(high=h, low=lo, close=c, window=ta_window, smooth_window=3)
    df["stoch_k"]      = stoch.stoch()
    df["stoch_d"]      = stoch.stoch_signal()
    df["roc"]          = ROCIndicator(close=c, window=ta_window).roc()
    df["williams_r"]   = WilliamsRIndicator(high=h, low=lo, close=c, lbp=ta_window).williams_r()
    df["awesome_osc"]  = AwesomeOscillatorIndicator(high=h, low=lo).awesome_oscillator()

    # Trend
    macd               = MACD(close=c)
    df["macd"]         = macd.macd()
    df["macd_signal"]  = macd.macd_signal()
    df["macd_diff"]    = macd.macd_diff()
    df["cci"]          = CCIIndicator(high=h, low=lo, close=c, window=ta_window).cci()
    adx                = ADXIndicator(high=h, low=lo, close=c, window=ta_window)
    df["adx"]          = adx.adx()
    df["adx_pos"]      = adx.adx_pos()
    df["adx_neg"]      = adx.adx_neg()
    aroon              = AroonIndicator(high=h, low=lo, window=ta_window)
    df["aroon_up"]     = aroon.aroon_up()
    df["aroon_down"]   = aroon.aroon_down()
    df["aroon_ind"]    = aroon.aroon_indicator()
    for w in (5, 10, 20, 50):
        df[f"ema_{w}"]          = EMAIndicator(close=c, window=w).ema_indicator()
        df[f"sma_{w}"]          = SMAIndicator(close=c, window=w).sma_indicator()
        df[f"close_vs_ema_{w}"] = c / df[f"ema_{w}"] - 1.0
        df[f"close_vs_sma_{w}"] = c / df[f"sma_{w}"] - 1.0
    df["ema5_vs_ema20"]  = df["ema_5"]  / df["ema_20"] - 1.0
    df["ema10_vs_ema50"] = df["ema_10"] / df["ema_50"] - 1.0

    # Volatility
    bb                   = BollingerBands(close=c, window=ta_window, window_dev=2)
    df["bb_upper"]       = bb.bollinger_hband()
    df["bb_lower"]       = bb.bollinger_lband()
    df["bb_mid"]         = bb.bollinger_mavg()
    df["bb_width"]       = bb.bollinger_wband()
    df["bb_pct"]         = bb.bollinger_pband()
    df["bb_hband_ind"]   = bb.bollinger_hband_indicator().astype("int8")
    df["bb_lband_ind"]   = bb.bollinger_lband_indicator().astype("int8")
    atr                  = AverageTrueRange(high=h, low=lo, close=c, window=ta_window)
    df["atr"]            = atr.average_true_range()
    df["atr_pct"]        = df["atr"] / c
    kc                   = KeltnerChannel(high=h, low=lo, close=c, window=ta_window)
    df["kc_upper"]       = kc.keltner_channel_hband()
    df["kc_lower"]       = kc.keltner_channel_lband()
    df["kc_mid"]         = kc.keltner_channel_mband()
    df["kc_pct"]         = kc.keltner_channel_pband()
    df["kc_wband"]       = kc.keltner_channel_wband()
    df["kc_hband_ind"]   = kc.keltner_channel_hband_indicator().astype("int8")
    df["kc_lband_ind"]   = kc.keltner_channel_lband_indicator().astype("int8")
    dc                   = DonchianChannel(high=h, low=lo, close=c, window=ta_window)
    df["dc_upper"]       = dc.donchian_channel_hband()
    df["dc_lower"]       = dc.donchian_channel_lband()
    df["dc_mid"]         = dc.donchian_channel_mband()
    df["dc_pct"]         = dc.donchian_channel_pband()
    df["dc_wband"]       = dc.donchian_channel_wband()
    df["ulcer_index"]    = UlcerIndex(close=c, window=ta_window).ulcer_index()

    # Volume
    df["obv"]            = OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()
    df["mfi"]            = MFIIndicator(high=h, low=lo, close=c, volume=v, window=ta_window).money_flow_index()
    df["cmf"]            = ChaikinMoneyFlowIndicator(high=h, low=lo, close=c, volume=v, window=ta_window).chaikin_money_flow()
    eom                  = EaseOfMovementIndicator(high=h, low=lo, volume=v, window=ta_window)
    df["eom"]            = eom.ease_of_movement()
    df["eom_signal"]     = eom.sma_ease_of_movement()
    vwap                 = VolumeWeightedAveragePrice(high=h, low=lo, close=c, volume=v, window=ta_window)
    df["vwap"]           = vwap.volume_weighted_average_price()
    df["close_vs_vwap"]  = c / df["vwap"] - 1.0

    return df


def _print_feature_report(
    *,
    n_cols_raw: int,
    n_new_features: int,
    n_cols_after_engineering: int,
    apply_supervised: bool,
    n_in: int,
    expected_supervised_cols: int | None,
    n_cols_supervised: int | None,
    target_col: str | None,
) -> None:
    sep = "=" * 62
    lines = [
        "",
        sep,
        "  Feature Engineering Report",
        sep,
        f"  Input columns (raw)         : {n_cols_raw}",
        f"  New engineered features      : {n_new_features}",
        f"  Total after engineering      : {n_cols_after_engineering}",
    ]
    if apply_supervised and expected_supervised_cols is not None:
        n_non_target = n_cols_after_engineering - 1
        check = "✓ OK" if expected_supervised_cols == n_cols_supervised else "✗ MISMATCH"
        lines += [
            f"  series_to_supervised n_in    : {n_in}",
            f"  Formula: {n_non_target} × (n_in={n_in}+1) + 1 target",
            f"  Expected supervised cols     : {expected_supervised_cols}",
            f"  Actual supervised cols       : {n_cols_supervised}  {check}",
        ]
    if target_col:
        lines.append(f"  Target column                : {target_col}  ← future data, label only!")
    lines.append(sep)
    print("\n".join(lines))
