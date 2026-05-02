"""5-minute OHLCV candle feature engineering for equity research.

Input columns: TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL

Dependencies:
    pip install pandas numpy ta

Feature count:
    Raw candle features:  ~60
    After shift(4) target proxy: +1
    TOTAL output columns: depends on ta windows, see build_sber_5m_candle_features.__doc__
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

# Feature groups used for reporting
_CANDLE_SHAPE_FEATURES = [
    "candle_body",
    "body_abs",
    "candle_range",
    "upper_wick",
    "lower_wick",
    "body_to_range",
    "upper_wick_to_range",
    "lower_wick_to_range",
    "direction",
    "is_green",
    "is_red",
    "is_doji_like",
    "close_pos_in_range",
    "open_pos_in_range",
    "range_pct_close",
    "range_pct_open",
]

_RETURN_FEATURES = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "gap_from_prev_close",
    "close_to_prev_high",
    "close_to_prev_low",
]

_VOLUME_FEATURES = [
    "vol_chg_1",
    "body_x_vol",
    "signed_vol",
    "ret1_x_volratio_6",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_sber_5m_candle_features(
    df: pd.DataFrame,
    *,
    windows: Iterable[int] = (3, 6, 12),
    ta_window: int = 14,
    shift: int = 4,
    add_target: bool = True,
    drop_invalid_rows: bool = False,
    copy: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build comprehensive candle + TA features for 5-minute OHLCV data.

    Expected columns::

        TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe with 5-minute OHLCV candles.
    windows : Iterable[int], default=(3, 6, 12)
        Rolling windows in bars.  For 5-minute data:
        3 = 15 min, 6 = 30 min, 12 = 60 min.
    ta_window : int, default=14
        Base period for TA indicators (RSI, ATR, ADX, etc.).
    shift : int, default=4
        Bars ahead for the forward-return target column ``target_ret_N``.
        Set to 0 or ``add_target=False`` to skip.
    add_target : bool, default=True
        Append a forward-return column shifted ``shift`` bars ahead.
        CAUTION: never use this column as a feature - it contains future data.
    drop_invalid_rows : bool, default=False
        Drop rows where OHLCV columns are NaN after numeric casting.
    copy : bool, default=True
        Work on a deep copy of the input (recommended).
    verbose : bool, default=True
        Print feature count report.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with engineered features.  All features use only
        current and past bar data - no look-ahead bias.

    Notes
    -----
    ta library:  pip install ta
    """
    _validate_columns(df)

    out = df.copy(deep=True) if copy else df

    out = _cast_numeric(out, drop_invalid_rows)
    out = _sort_by_time(out)

    n_cols_before = out.shape[1]

    # -----------------------------------------------------------------------
    # 1. Candle geometry
    # -----------------------------------------------------------------------
    out = _add_candle_shape(out)

    # -----------------------------------------------------------------------
    # 2. Returns / momentum
    # -----------------------------------------------------------------------
    out = _add_returns(out)

    # -----------------------------------------------------------------------
    # 3. Rolling context (price + range + volume)
    # -----------------------------------------------------------------------
    out = _add_rolling_context(out, list(windows))

    # -----------------------------------------------------------------------
    # 4. Volume interaction
    # -----------------------------------------------------------------------
    out = _add_volume_features(out)

    # -----------------------------------------------------------------------
    # 5. TA indicators
    # -----------------------------------------------------------------------
    if _TA_AVAILABLE:
        out = _add_ta_indicators(out, ta_window=ta_window)
    else:  # pragma: no cover
        import warnings
        warnings.warn(
            "ta library not found.  Install with: pip install ta\n"
            "TA indicators will be skipped.",
            ImportWarning,
            stacklevel=2,
        )

    # -----------------------------------------------------------------------
    # 6. Forward-return target (shift=4 => +20 min)
    # -----------------------------------------------------------------------
    if add_target and shift > 0:
        out[f"target_ret_{shift}"] = out["CLOSE"].pct_change(shift).shift(-shift)

    n_cols_after = out.shape[1]
    n_new = n_cols_after - n_cols_before

    if verbose:
        _print_feature_report(
            n_cols_before=n_cols_before,
            n_cols_after=n_cols_after,
            n_new=n_new,
            shift=shift,
            add_target=add_target,
        )

    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_columns(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _cast_numeric(
    df: pd.DataFrame,
    drop_invalid: bool,
) -> pd.DataFrame:
    numeric_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if drop_invalid:
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return df


def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    if not {"DATE", "TIME"}.issubset(df.columns):
        return df
    date_str = df["DATE"].astype(str).str.strip()
    time_str = df["TIME"].astype(str).str.zfill(6).str.strip()
    df["datetime"] = pd.to_datetime(
        date_str + time_str, format="%Y%m%d%H%M%S", errors="coerce"
    )
    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)
    return df


def _add_candle_shape(df: pd.DataFrame) -> pd.DataFrame:
    rng = df["HIGH"] - df["LOW"]
    safe_rng = rng.where(rng.abs() > _EPS, np.nan)
    max_oc = np.maximum(df["OPEN"], df["CLOSE"])
    min_oc = np.minimum(df["OPEN"], df["CLOSE"])

    df["candle_body"] = df["CLOSE"] - df["OPEN"]
    df["body_abs"] = df["candle_body"].abs()
    df["candle_range"] = rng
    df["upper_wick"] = df["HIGH"] - max_oc
    df["lower_wick"] = min_oc - df["LOW"]

    df["body_to_range"] = (df["body_abs"] / safe_rng).clip(0.0, 1.0)
    df["upper_wick_to_range"] = (df["upper_wick"] / safe_rng).clip(0.0, 1.0)
    df["lower_wick_to_range"] = (df["lower_wick"] / safe_rng).clip(0.0, 1.0)

    df["direction"] = np.sign(df["candle_body"]).astype("float64")
    df["is_green"] = (df["CLOSE"] > df["OPEN"]).astype("int8")
    df["is_red"] = (df["CLOSE"] < df["OPEN"]).astype("int8")
    df["is_doji_like"] = (df["body_to_range"] <= 0.10).astype("int8")

    df["close_pos_in_range"] = ((df["CLOSE"] - df["LOW"]) / safe_rng).clip(0.0, 1.0)
    df["open_pos_in_range"] = ((df["OPEN"] - df["LOW"]) / safe_rng).clip(0.0, 1.0)

    df["range_pct_close"] = df["candle_range"] / df["CLOSE"].replace(0.0, np.nan)
    df["range_pct_open"] = df["candle_range"] / df["OPEN"].replace(0.0, np.nan)
    return df


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["CLOSE"].shift(1)
    prev_high = df["HIGH"].shift(1)
    prev_low = df["LOW"].shift(1)

    df["ret_1"] = df["CLOSE"].pct_change(1)
    df["ret_3"] = df["CLOSE"].pct_change(3)
    df["ret_6"] = df["CLOSE"].pct_change(6)
    df["ret_12"] = df["CLOSE"].pct_change(12)

    df["gap_from_prev_close"] = df["OPEN"] / prev_close - 1.0
    df["close_to_prev_high"] = df["CLOSE"] / prev_high - 1.0
    df["close_to_prev_low"] = df["CLOSE"] / prev_low - 1.0
    return df


def _add_rolling_context(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"ret_{w}"] = df["CLOSE"].pct_change(w)
        df[f"range_mean_{w}"] = df["candle_range"].rolling(w, min_periods=w).mean()
        df[f"range_std_{w}"] = df["candle_range"].rolling(w, min_periods=w).std()
        df[f"body_abs_mean_{w}"] = df["body_abs"].rolling(w, min_periods=w).mean()

        df[f"vol_ma_{w}"] = df["VOL"].rolling(w, min_periods=w).mean()
        df[f"vol_std_{w}"] = df["VOL"].rolling(w, min_periods=w).std()
        df[f"vol_ratio_{w}"] = df["VOL"] / df[f"vol_ma_{w}"]

        df[f"close_ma_{w}"] = df["CLOSE"].rolling(w, min_periods=w).mean()
        df[f"close_std_{w}"] = df["CLOSE"].rolling(w, min_periods=w).std()
        df[f"close_vs_ma_{w}"] = df["CLOSE"] / df[f"close_ma_{w}"] - 1.0

        safe_range_std = df[f"range_std_{w}"].replace(0.0, np.nan)
        safe_close_std = df[f"close_std_{w}"].replace(0.0, np.nan)
        df[f"range_zscore_{w}"] = (
            (df["candle_range"] - df[f"range_mean_{w}"]) / safe_range_std
        )
        df[f"close_zscore_{w}"] = (
            (df["CLOSE"] - df[f"close_ma_{w}"]) / safe_close_std
        )
    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_chg_1"] = df["VOL"].pct_change(1)
    df["body_x_vol"] = df["body_abs"] * df["VOL"]
    df["signed_vol"] = df["direction"] * df["VOL"]
    vol_ratio_6 = df.get("vol_ratio_6", pd.Series(np.nan, index=df.index))
    df["ret1_x_volratio_6"] = df["ret_1"] * vol_ratio_6
    return df


def _add_ta_indicators(df: pd.DataFrame, ta_window: int = 14) -> pd.DataFrame:
    c = df["CLOSE"]
    h = df["HIGH"]
    lo = df["LOW"]
    v = df["VOL"]

    # --- Momentum -----------------------------------------------------------
    rsi = RSIIndicator(close=c, window=ta_window)
    df["rsi"] = rsi.rsi()

    stoch = StochasticOscillator(high=h, low=lo, close=c, window=ta_window, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    roc = ROCIndicator(close=c, window=ta_window)
    df["roc"] = roc.roc()

    wr = WilliamsRIndicator(high=h, low=lo, close=c, lbp=ta_window)
    df["williams_r"] = wr.williams_r()

    ao = AwesomeOscillatorIndicator(high=h, low=lo)
    df["awesome_osc"] = ao.awesome_oscillator()

    # --- Trend --------------------------------------------------------------
    macd = MACD(close=c)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    cci = CCIIndicator(high=h, low=lo, close=c, window=ta_window)
    df["cci"] = cci.cci()

    adx = ADXIndicator(high=h, low=lo, close=c, window=ta_window)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    aroon = AroonIndicator(high=h, low=lo, window=ta_window)
    df["aroon_up"] = aroon.aroon_up()
    df["aroon_down"] = aroon.aroon_down()
    df["aroon_ind"] = aroon.aroon_indicator()

    for w in (5, 10, 20, 50):
        ema = EMAIndicator(close=c, window=w)
        sma = SMAIndicator(close=c, window=w)
        df[f"ema_{w}"] = ema.ema_indicator()
        df[f"sma_{w}"] = sma.sma_indicator()
        df[f"close_vs_ema_{w}"] = c / df[f"ema_{w}"] - 1.0
        df[f"close_vs_sma_{w}"] = c / df[f"sma_{w}"] - 1.0

    # EMA crossover signals
    df["ema5_vs_ema20"] = df["ema_5"] / df["ema_20"] - 1.0
    df["ema10_vs_ema50"] = df["ema_10"] / df["ema_50"] - 1.0

    # --- Volatility ---------------------------------------------------------
    bb = BollingerBands(close=c, window=ta_window, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()
    df["bb_hband_ind"] = bb.bollinger_hband_indicator().astype("int8")
    df["bb_lband_ind"] = bb.bollinger_lband_indicator().astype("int8")

    atr = AverageTrueRange(high=h, low=lo, close=c, window=ta_window)
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / c

    kc = KeltnerChannel(high=h, low=lo, close=c, window=ta_window)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()
    df["kc_mid"] = kc.keltner_channel_mband()
    df["kc_pct"] = kc.keltner_channel_pband()
    df["kc_wband"] = kc.keltner_channel_wband()
    df["kc_hband_ind"] = kc.keltner_channel_hband_indicator().astype("int8")
    df["kc_lband_ind"] = kc.keltner_channel_lband_indicator().astype("int8")

    dc = DonchianChannel(high=h, low=lo, close=c, window=ta_window)
    df["dc_upper"] = dc.donchian_channel_hband()
    df["dc_lower"] = dc.donchian_channel_lband()
    df["dc_mid"] = dc.donchian_channel_mband()
    df["dc_pct"] = dc.donchian_channel_pband()
    df["dc_wband"] = dc.donchian_channel_wband()

    ui = UlcerIndex(close=c, window=ta_window)
    df["ulcer_index"] = ui.ulcer_index()

    # --- Volume -------------------------------------------------------------
    obv = OnBalanceVolumeIndicator(close=c, volume=v)
    df["obv"] = obv.on_balance_volume()

    mfi = MFIIndicator(high=h, low=lo, close=c, volume=v, window=ta_window)
    df["mfi"] = mfi.money_flow_index()

    cmf = ChaikinMoneyFlowIndicator(high=h, low=lo, close=c, volume=v, window=ta_window)
    df["cmf"] = cmf.chaikin_money_flow()

    eom = EaseOfMovementIndicator(high=h, low=lo, volume=v, window=ta_window)
    df["eom"] = eom.ease_of_movement()
    df["eom_signal"] = eom.sma_ease_of_movement()

    vwap = VolumeWeightedAveragePrice(high=h, low=lo, close=c, volume=v, window=ta_window)
    df["vwap"] = vwap.volume_weighted_average_price()
    df["close_vs_vwap"] = c / df["vwap"] - 1.0

    return df


def _print_feature_report(
    *,
    n_cols_before: int,
    n_cols_after: int,
    n_new: int,
    shift: int,
    add_target: bool,
) -> None:
    target_note = (
        f"  ├── incl. target_ret_{shift} (future data - DO NOT use as feature!)"
        if add_target and shift > 0
        else ""
    )
    print(
        "\n" + "=" * 60 + "\n"
        f"  Feature Engineering Report\n"
        + "=" * 60 + "\n"
        f"  Columns before  : {n_cols_before}\n"
        f"  New features    : {n_new}\n"
        f"  Columns after   : {n_cols_after}\n"
        + (target_note + "\n" if target_note else "")
        + "=" * 60 + "\n"
    )
