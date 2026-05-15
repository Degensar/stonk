import argparse
import io
import math
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


@dataclass
class VCPConfig:
    min_price: float = 10.0
    min_avg_dollar_volume: float = 20_000_000.0
    min_rs_rank: float = 80.0
    max_base_depth_pct: float = 35.0
    max_last_contraction_pct: float = 12.0
    max_pivot_distance_pct: float = 5.0
    min_contractions: int = 2
    require_volume_dry_up: bool = True


@dataclass
class VCPResult:
    ticker: str
    close: float
    rs_rank: float
    base_depth_pct: float
    contraction_count: int
    contractions_pct: str
    pivot_price: float
    pivot_distance_pct: float
    avg_dollar_volume: float
    volume_dry_up_ratio: float
    score: float


@dataclass
class PrefilterStats:
    downloaded: int = 0
    usable_history: int = 0
    sepa_prefilter_pass: int = 0
    rs_rank_pass: int = 0
    download_failures: int = 0


def fetch_us_market_universe(include_etfs: bool = False) -> list[str]:
    """Fetch active US-listed symbols from Nasdaq Trader symbol directories."""
    nasdaq = _read_nasdaq_trader_file(NASDAQ_LISTED_URL)
    other = _read_nasdaq_trader_file(OTHER_LISTED_URL)

    symbols: list[tuple[str, str, str]] = []

    for _, row in nasdaq.iterrows():
        symbols.append(
            (
                str(row.get("Symbol", "")),
                str(row.get("Security Name", "")),
                str(row.get("ETF", "N")),
            )
        )

    for _, row in other.iterrows():
        symbols.append(
            (
                str(row.get("ACT Symbol", "")),
                str(row.get("Security Name", "")),
                str(row.get("ETF", "N")),
            )
        )

    clean_symbols = []
    seen = set()
    for symbol, name, etf_flag in symbols:
        if not include_etfs and etf_flag.upper() == "Y":
            continue
        if not _looks_like_operating_company(symbol, name):
            continue

        yf_symbol = symbol.replace(".", "-").strip().upper()
        if yf_symbol and yf_symbol not in seen:
            seen.add(yf_symbol)
            clean_symbols.append(yf_symbol)

    return sorted(clean_symbols)


def _read_nasdaq_trader_file(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")

    lines = [line for line in text.splitlines() if not line.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")


def _looks_like_operating_company(symbol: str, name: str) -> bool:
    symbol = symbol.strip().upper()
    name_lower = name.lower()

    if not symbol or symbol in {"NAN", "NULL"}:
        return False
    if any(char in symbol for char in ("$", "^", "/", " ")):
        return False

    excluded_name_parts = (
        "warrant",
        "right",
        "unit",
        "preferred",
        "preference",
        "depositary",
        "depositary share",
        "notes due",
        "senior notes",
        "subordinated notes",
        "debenture",
        "bond",
        "etf",
        "etn",
        "fund",
        "trust",
    )
    return not any(part in name_lower for part in excluded_name_parts)


def download_history(
    tickers: list[str],
    period: str,
    interval: str,
    batch_size: int,
    pause: float,
    request_timeout: float = 30,
    retries: int = 1,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start : start + batch_size]
        print(f"Downloading {start + 1:,}-{start + len(batch):,} of {len(tickers):,}")

        data = download_batch(batch, period, interval, request_timeout, retries)

        if data.empty:
            time.sleep(pause)
            continue

        histories.update(extract_batch_histories(data, batch))

        time.sleep(pause)

    return histories


def download_batch(
    batch: list[str],
    period: str,
    interval: str,
    request_timeout: float,
    retries: int,
) -> pd.DataFrame:
    last_error = None

    for attempt in range(retries + 1):
        try:
            return yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
                timeout=request_timeout,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1 + attempt)

    print(f"Download failed for batch starting {batch[0]}: {last_error}")
    return pd.DataFrame()


def extract_batch_histories(data: pd.DataFrame, batch: list[str]) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}

    if isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        for ticker in batch:
            if ticker in available:
                frame = data[ticker].dropna(how="all")
                if not frame.empty:
                    histories[ticker] = frame
    elif len(batch) == 1:
        histories[batch[0]] = data.dropna(how="all")

    return histories


def download_sepa_prefiltered_histories(
    tickers: list[str],
    period: str,
    interval: str,
    batch_size: int,
    pause: float,
    config: VCPConfig,
    request_timeout: float,
    retries: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], PrefilterStats]:
    histories: dict[str, pd.DataFrame] = {}
    raw_rs_returns: dict[str, float] = {}
    stats = PrefilterStats()

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start : start + batch_size]
        print(f"Downloading {start + 1:,}-{start + len(batch):,} of {len(tickers):,}")
        stats.downloaded += len(batch)

        data = download_batch(batch, period, interval, request_timeout, retries)
        if data.empty:
            stats.download_failures += len(batch)
            time.sleep(pause)
            continue

        batch_histories = extract_batch_histories(data, batch)
        for ticker, frame in batch_histories.items():
            prepared = prepare_price_frame(frame)
            if prepared is None:
                continue

            stats.usable_history += 1
            close = prepared["Close"]
            weighted_return = _weighted_relative_strength_return(close)
            if weighted_return is not None and math.isfinite(weighted_return):
                raw_rs_returns[ticker] = weighted_return

            if passes_sepa_prefilter(prepared, config):
                histories[ticker] = prepared
                stats.sepa_prefilter_pass += 1

        print(
            "  SEPA prefilter kept "
            f"{stats.sepa_prefilter_pass:,}/{stats.usable_history:,} usable symbols so far"
        )
        time.sleep(pause)

    return histories, raw_rs_returns, stats


def rank_rs_returns(raw_rs_returns: dict[str, float]) -> dict[str, float]:
    if not raw_rs_returns:
        return {}

    ranked = pd.Series(raw_rs_returns).rank(pct=True) * 100
    return ranked.to_dict()


def compute_rs_ranks(histories: dict[str, pd.DataFrame]) -> dict[str, float]:
    returns = {}
    for ticker, frame in histories.items():
        close = _column(frame, "Close")
        if close is None or len(close.dropna()) < 130:
            continue

        weighted_return = _weighted_relative_strength_return(close.dropna())
        if weighted_return is not None and math.isfinite(weighted_return):
            returns[ticker] = weighted_return

    if not returns:
        return {}

    ranked = pd.Series(returns).rank(pct=True) * 100
    return ranked.to_dict()


def prepare_price_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    close = _column(frame, "Close")
    high = _column(frame, "High")
    low = _column(frame, "Low")
    volume = _column(frame, "Volume")

    if close is None or high is None or low is None or volume is None:
        return None

    data = pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume}).dropna()
    if len(data) < 260:
        return None

    return data


def passes_sepa_prefilter(data: pd.DataFrame, config: VCPConfig) -> bool:
    close = data["Close"]
    volume = data["Volume"]

    last_close = float(close.iloc[-1])
    if last_close < config.min_price:
        return False

    avg_dollar_volume = float((close.iloc[-50:] * volume.iloc[-50:]).mean())
    if avg_dollar_volume < config.min_avg_dollar_volume:
        return False

    return passes_trend_template(close)


def _weighted_relative_strength_return(close: pd.Series) -> float | None:
    windows = (63, 126, 189, 252)
    weights = (0.4, 0.2, 0.2, 0.2)

    score = 0.0
    used_weight = 0.0
    for window, weight in zip(windows, weights):
        if len(close) <= window:
            continue
        start = close.iloc[-window]
        end = close.iloc[-1]
        if start > 0:
            score += ((end / start) - 1.0) * weight
            used_weight += weight

    if used_weight == 0:
        return None
    return score / used_weight


def scan_vcp(
    histories: dict[str, pd.DataFrame],
    config: VCPConfig,
    rs_ranks: dict[str, float],
) -> list[VCPResult]:
    results = []

    for ticker, frame in histories.items():
        result = analyze_ticker(ticker, frame, config, rs_ranks.get(ticker))
        if result is not None:
            results.append(result)

    return sorted(results, key=lambda item: item.score, reverse=True)


def analyze_ticker(
    ticker: str,
    frame: pd.DataFrame,
    config: VCPConfig,
    rs_rank: float | None,
) -> VCPResult | None:
    data = prepare_price_frame(frame)
    if data is None:
        return None

    close = data["Close"]
    volume = data["Volume"]

    last_close = float(close.iloc[-1])
    if last_close < config.min_price:
        return None

    avg_dollar_volume = float((close.iloc[-50:] * volume.iloc[-50:]).mean())
    if avg_dollar_volume < config.min_avg_dollar_volume:
        return None

    if rs_rank is None or rs_rank < config.min_rs_rank:
        return None

    if not passes_trend_template(close):
        return None

    base = data.iloc[-65:]
    base_high = float(base["High"].max())
    base_low = float(base["Low"].min())
    base_depth_pct = _pct_down(base_high, base_low)
    if base_depth_pct > config.max_base_depth_pct:
        return None

    contractions = find_contractions(base)
    if len(contractions) < config.min_contractions:
        return None
    if not contractions_are_tightening(contractions):
        return None

    contraction_pcts = [_pct_down(high_value, low_value) for high_value, low_value in contractions]
    last_contraction_pct = contraction_pcts[-1]
    if last_contraction_pct > config.max_last_contraction_pct:
        return None

    pivot_price = float(base["High"].iloc[-20:].max())
    pivot_distance_pct = ((pivot_price - last_close) / pivot_price) * 100
    if pivot_distance_pct < -2 or pivot_distance_pct > config.max_pivot_distance_pct:
        return None

    volume_dry_up_ratio = float(volume.iloc[-10:].mean() / volume.iloc[-50:].mean())
    if config.require_volume_dry_up and volume_dry_up_ratio > 0.75:
        return None

    score = (
        rs_rank
        - base_depth_pct
        - last_contraction_pct
        - max(pivot_distance_pct, 0)
        + max(0, (1.0 - volume_dry_up_ratio) * 20)
        + len(contractions) * 2
    )

    return VCPResult(
        ticker=ticker,
        close=round(last_close, 2),
        rs_rank=round(float(rs_rank), 1),
        base_depth_pct=round(base_depth_pct, 1),
        contraction_count=len(contractions),
        contractions_pct=", ".join(f"{value:.1f}" for value in contraction_pcts),
        pivot_price=round(pivot_price, 2),
        pivot_distance_pct=round(pivot_distance_pct, 1),
        avg_dollar_volume=round(avg_dollar_volume, 0),
        volume_dry_up_ratio=round(volume_dry_up_ratio, 2),
        score=round(score, 1),
    )


def passes_trend_template(close: pd.Series) -> bool:
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()

    if pd.isna(sma50.iloc[-1]) or pd.isna(sma150.iloc[-1]) or pd.isna(sma200.iloc[-1]):
        return False

    last = close.iloc[-1]
    high_52w = close.iloc[-252:].max()
    low_52w = close.iloc[-252:].min()

    return all(
        (
            last > sma50.iloc[-1],
            last > sma150.iloc[-1],
            last > sma200.iloc[-1],
            sma50.iloc[-1] > sma150.iloc[-1],
            sma150.iloc[-1] > sma200.iloc[-1],
            sma200.iloc[-1] > sma200.iloc[-22],
            last >= high_52w * 0.75,
            last >= low_52w * 1.30,
        )
    )


def find_contractions(base: pd.DataFrame) -> list[tuple[float, float]]:
    """Approximate VCP contractions by alternating local highs and later lows."""
    close = base["Close"]
    high = base["High"]
    low = base["Low"]

    swing_highs = _swing_points(high, mode="high")
    swing_lows = _swing_points(low, mode="low")
    swings = sorted([(idx, "H") for idx in swing_highs] + [(idx, "L") for idx in swing_lows])

    contractions = []
    active_high_idx = None

    for idx, swing_type in swings:
        if swing_type == "H":
            if active_high_idx is None or high.iloc[idx] > high.iloc[active_high_idx]:
                active_high_idx = idx
        elif active_high_idx is not None and idx > active_high_idx:
            high_value = float(high.iloc[active_high_idx])
            low_value = float(low.iloc[idx])
            if low_value < high_value and close.iloc[idx] >= close.iloc[0] * 0.75:
                contractions.append((high_value, low_value))
                active_high_idx = None

    if len(contractions) < 2:
        first_high = float(high.iloc[:20].max())
        first_low = float(low.iloc[:35].min())
        last_high = float(high.iloc[-25:].max())
        last_low = float(low.iloc[-15:].min())
        contractions = [(first_high, first_low), (last_high, last_low)]

    return contractions[-4:]


def _swing_points(series: pd.Series, mode: str, left: int = 3, right: int = 3) -> list[int]:
    points = []
    values = series.reset_index(drop=True)

    for idx in range(left, len(values) - right):
        window = values.iloc[idx - left : idx + right + 1]
        value = values.iloc[idx]
        if mode == "high" and value == window.max():
            points.append(idx)
        elif mode == "low" and value == window.min():
            points.append(idx)

    return points


def contractions_are_tightening(contractions: list[tuple[float, float]]) -> bool:
    contraction_pcts = [_pct_down(high_value, low_value) for high_value, low_value in contractions]
    if len(contraction_pcts) < 2:
        return False

    tightening_steps = sum(
        current <= previous * 1.10
        for previous, current in zip(contraction_pcts, contraction_pcts[1:])
    )
    return tightening_steps >= len(contraction_pcts) - 1


def _pct_down(high_value: float, low_value: float) -> float:
    if high_value <= 0:
        return 100.0
    return ((high_value - low_value) / high_value) * 100


def _column(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if name in frame.columns:
        return frame[name]
    lower_map = {str(column).lower(): column for column in frame.columns}
    column = lower_map.get(name.lower())
    if column is None:
        return None
    return frame[column]


def read_tickers(path: str | None, include_etfs: bool, limit: int | None) -> list[str]:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            tickers = [
                line.strip().upper().replace(".", "-")
                for line in handle
                if line.strip() and not line.startswith("#")
            ]
    else:
        tickers = fetch_us_market_universe(include_etfs=include_etfs)

    if limit is not None:
        tickers = tickers[:limit]
    return tickers


def write_results(results: Iterable[VCPResult], output: str) -> pd.DataFrame:
    rows = [asdict(result) for result in results]
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan US stocks for a Minervini-style VCP setup."
    )
    parser.add_argument("--tickers-file", help="Optional newline-delimited ticker universe.")
    parser.add_argument("--include-etfs", action="store_true", help="Include ETFs in the scan.")
    parser.add_argument("--limit", type=int, help="Scan only the first N symbols, useful for testing.")
    parser.add_argument("--period", default="18mo", help="History period passed to yfinance.")
    parser.add_argument("--interval", default="1d", help="History interval passed to yfinance.")
    parser.add_argument("--batch-size", type=int, default=50, help="Symbols per yfinance request.")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds to pause between batches.")
    parser.add_argument("--request-timeout", type=float, default=30, help="Seconds before a yfinance batch times out.")
    parser.add_argument("--retries", type=int, default=1, help="Retry count for failed yfinance batches.")
    parser.add_argument("--output", default="vcp_candidates.csv", help="CSV output file.")
    parser.add_argument("--top", type=int, default=50, help="Rows to print.")
    parser.add_argument("--min-price", type=float, default=VCPConfig.min_price)
    parser.add_argument("--min-dollar-volume", type=float, default=VCPConfig.min_avg_dollar_volume)
    parser.add_argument("--min-rs-rank", type=float, default=VCPConfig.min_rs_rank)
    parser.add_argument("--max-base-depth", type=float, default=VCPConfig.max_base_depth_pct)
    parser.add_argument("--max-last-contraction", type=float, default=VCPConfig.max_last_contraction_pct)
    parser.add_argument("--max-pivot-distance", type=float, default=VCPConfig.max_pivot_distance_pct)
    parser.add_argument("--min-contractions", type=int, default=VCPConfig.min_contractions)
    parser.add_argument("--allow-normal-volume", action="store_true", help="Disable volume dry-up filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VCPConfig(
        min_price=args.min_price,
        min_avg_dollar_volume=args.min_dollar_volume,
        min_rs_rank=args.min_rs_rank,
        max_base_depth_pct=args.max_base_depth,
        max_last_contraction_pct=args.max_last_contraction,
        max_pivot_distance_pct=args.max_pivot_distance,
        min_contractions=args.min_contractions,
        require_volume_dry_up=not args.allow_normal_volume,
    )

    tickers = read_tickers(args.tickers_file, args.include_etfs, args.limit)
    print(f"Universe size: {len(tickers):,}")
    print(f"Config: {config}")

    histories, raw_rs_returns, stats = download_sepa_prefiltered_histories(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        batch_size=args.batch_size,
        pause=args.pause,
        config=config,
        request_timeout=args.request_timeout,
        retries=args.retries,
    )

    rs_ranks = rank_rs_returns(raw_rs_returns)
    rs_filtered_histories = {
        ticker: frame
        for ticker, frame in histories.items()
        if rs_ranks.get(ticker, 0) >= config.min_rs_rank
    }
    stats.rs_rank_pass = len(rs_filtered_histories)

    print("\nFirst-pass filters:")
    print(f"  Download attempted: {stats.downloaded:,}")
    print(f"  Usable histories:   {stats.usable_history:,}")
    print(f"  SEPA MA/liquidity:  {stats.sepa_prefilter_pass:,}")
    print(f"  RS rank >= {config.min_rs_rank:g}: {stats.rs_rank_pass:,}")
    if stats.download_failures:
        print(f"  Batch misses:       {stats.download_failures:,}")

    results = scan_vcp(rs_filtered_histories, config, rs_ranks)
    table = write_results(results, args.output)

    print(f"\nFound {len(results):,} candidates.")
    if table.empty:
        print("No candidates passed. Try relaxing --min-rs-rank, --max-pivot-distance, or --allow-normal-volume.")
    else:
        print(table.head(args.top).to_string(index=False))
    print(f"\nSaved: {args.output}")
    print("This is a screening tool, not financial advice. Confirm charts, earnings, liquidity, and risk manually.")


if __name__ == "__main__":
    main()
