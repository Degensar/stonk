#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from futu import (
    CustomIndicatorFilter,
    KLType,
    Market,
    OpenQuoteContext,
    RET_OK,
    RelativePosition,
    SimpleFilter,
    StockField,
)


DEFAULT_MIN_MARKET_CAP = 100_000_000.0


def indicator_more(field1, field2, para1=None, para2=None) -> CustomIndicatorFilter:
    filter_item = CustomIndicatorFilter()
    filter_item.stock_field1 = field1
    filter_item.stock_field2 = field2
    filter_item.stock_field1_para = para1 or []
    filter_item.stock_field2_para = para2 or []
    filter_item.relative_position = RelativePosition.MORE
    filter_item.ktype = KLType.K_DAY
    filter_item.is_no_filter = False
    return filter_item


def simple_min(field, minimum: float) -> SimpleFilter:
    filter_item = SimpleFilter()
    filter_item.stock_field = field
    filter_item.filter_min = minimum
    filter_item.is_no_filter = False
    return filter_item


def build_filters(
    min_52w_low_pct: float,
    max_below_52w_high_pct: float | None,
    min_market_cap: float | None,
) -> list:
    filters = [
        indicator_more(StockField.PRICE, StockField.MA, para2=[50]),
        indicator_more(StockField.MA, StockField.MA, para1=[50], para2=[150]),
        indicator_more(StockField.MA, StockField.MA, para1=[150], para2=[200]),
        # FUTU SimpleFilter min/max are inclusive, so nudge above 30 for "more than 30%".
        simple_min(StockField.CUR_PRICE_TO_LOWEST52_WEEKS_RATIO, min_52w_low_pct),
    ]
    if min_market_cap is not None and min_market_cap > 0:
        filters.append(simple_min(StockField.MARKET_VAL, min_market_cap))
    if max_below_52w_high_pct is not None:
        filters.append(
            simple_min(
                StockField.CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO,
                -abs(max_below_52w_high_pct),
            )
        )
    return filters


def fetch_filtered_stocks(
    host: str,
    port: int,
    filters: list,
    page_size: int,
) -> pd.DataFrame:
    rows = []
    quote_ctx = OpenQuoteContext(host=host, port=port)
    try:
        begin = 0
        while True:
            ret, payload = quote_ctx.get_stock_filter(
                market=Market.US,
                filter_list=filters,
                begin=begin,
                num=page_size,
            )
            if ret != RET_OK:
                raise RuntimeError(f"get_stock_filter failed at begin={begin}: {payload}")

            last_page, all_count, items = payload
            print(f"Fetched {begin + len(items):,}/{all_count:,}")

            for item in items:
                data = item.__dict__
                price = data.get(("price", "k_day"))
                ma50 = data.get(("ma", "50", "k_day"))
                ma150 = data.get(("ma", "150", "k_day"))
                ma200 = data.get(("ma", "200", "k_day"))
                low_ratio = data.get("cur_price_to_lowest52_weeks_ratio")
                high_ratio = data.get("cur_price_to_highest52_weeks_ratio")
                market_cap = data.get("market_val")

                estimated_low = (
                    price / (1 + low_ratio / 100)
                    if price is not None and low_ratio is not None and low_ratio > -100
                    else None
                )
                estimated_high = (
                    price / (1 + high_ratio / 100)
                    if price is not None and high_ratio is not None and high_ratio > -100
                    else None
                )

                rows.append(
                    {
                        "code": item.stock_code,
                        "ticker": item.stock_code.removeprefix("US."),
                        "name": item.stock_name,
                        "price": price,
                        "ma50": ma50,
                        "ma150": ma150,
                        "ma200": ma200,
                        "market_cap": market_cap,
                        "price_to_52w_low_pct": low_ratio,
                        "price_to_52w_high_pct": high_ratio,
                        "estimated_52w_low": estimated_low,
                        "estimated_52w_high": estimated_high,
                        "price_above_ma50_pct": ((price / ma50 - 1) * 100) if price and ma50 else None,
                        "ma50_above_ma150_pct": ((ma50 / ma150 - 1) * 100) if ma50 and ma150 else None,
                        "ma150_above_ma200_pct": ((ma150 / ma200 - 1) * 100) if ma150 and ma200 else None,
                        "futu_filter_price_gt_ma50": True,
                        "futu_filter_ma50_gt_ma150": True,
                        "futu_filter_ma150_gt_ma200": True,
                        "futu_filter_gt_30_pct_above_52w_low": True,
                        "futu_filter_market_cap_gte_min": market_cap is not None,
                        "futu_filter_within_25_pct_of_52w_high": high_ratio is not None,
                    }
                )

            if last_page or not items:
                break
            begin += page_size
    finally:
        quote_ctx.close()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ticker", kind="stable")


def parse_args() -> argparse.Namespace:
    default_output = f"data/futu_get_stock_filter_sepa_ma_52wlow_{datetime.now():%Y%m%d_%H%M%S}.csv"
    parser = argparse.ArgumentParser(
        description="Export a fresh FUTU get_stock_filter SEPA/MA prefilter CSV for the US market."
    )
    parser.add_argument("--host", default="127.0.0.1", help="FUTU OpenD host.")
    parser.add_argument("--port", type=int, default=11111, help="FUTU OpenD port.")
    parser.add_argument("--page-size", type=int, default=200, help="Rows per get_stock_filter page.")
    parser.add_argument("--output", default=default_output, help="Output CSV path.")
    parser.add_argument(
        "--min-52w-low-pct",
        type=float,
        default=30.000001,
        help="Minimum current-price-to-52-week-low percent. Default is just above 30.",
    )
    parser.add_argument(
        "--max-below-52w-high-pct",
        type=float,
        default=25,
        help="Require price to be no more than this percent below the 52-week high. Use -1 to disable.",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=DEFAULT_MIN_MARKET_CAP,
        help=(
            "Minimum market capitalization for US stocks, in US dollars. "
            "Default is 100,000,000. Use 0 to disable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_below_high = None if args.max_below_52w_high_pct < 0 else args.max_below_52w_high_pct
    filters = build_filters(
        min_52w_low_pct=args.min_52w_low_pct,
        max_below_52w_high_pct=max_below_high,
        min_market_cap=args.min_market_cap,
    )

    result = fetch_filtered_stocks(
        host=args.host,
        port=args.port,
        filters=filters,
        page_size=args.page_size,
    )

    output = Path(args.output)
    result.to_csv(output, index=False, encoding="utf-8-sig", float_format="%.10g")
    print(f"Wrote {len(result):,} rows to {output}")


if __name__ == "__main__":
    main()
