"""Cleaning + forecasting logic for the weekly ordering helper (v1)."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from parser import load_payhere

# ---------- Settings ----------
THU_SUN = ["Thursday", "Friday", "Saturday", "Sunday"]
OLD_END = "2026-01-01"             # old location data ends before this
NEW_LOCATION_START = "2026-09-01"  # new location data starts here

# Irregular days (reserved / special occasions): excluded from the forecast
EVENT_CATEGORIES = [
    "abc market 🧺", "janemarch mkt", "25.04.26 제인마치",
    "sunday heaven", "🦪3.10 sunday🌿",
    "pot 3주년", "🌷jenny 시음회", "viaplain🧡💛", "타르틴", "🌿green j", "⚽️니벨크랙🍷",
    "10/11(대관)", "daycare center", "wendys 5yrs",
    "❄️12월 pot", "🔴 red abc",
]
# Rows in these categories hold unlabeled products; classify them by name
CLASSIFY_CATS = ["❄️12월", "❄️12월 pot", "🎄", "🔴 red abc"]
WINE_OVERRIDE = ["daily glass", "half bottle", "half", "tmb half", "chillable red",
                 "acid frick", "carbonic maceration"]
FOOD_FIX = ["pappardelle", "cheese olives", "spicy pepperoni", "charcueterie", "snowman", "onion ring"]
ALIASES = {"chicken wig": "chicken wing"}
BOTTLE_PRICE = 80000


def norm(name):
    s = str(name).lower().strip()
    s = re.sub(r"^[>+\s]+", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build_dataset(data_dir):
    """Load every export in data_dir, clean and label. Returns (all_data, problems)."""
    problems, frames = [], []
    for f in sorted(Path(data_dir).glob("*.xlsx")):
        if f.name.startswith("~$"):          # Excel temp file
            continue
        try:
            df = load_payhere(f)
        except Exception as e:               # bad file: report it, keep going
            problems.append(f"{f.name}: {type(e).__name__}: {e}")
            continue
        df["source_file"] = f.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No readable .xlsx exports in {data_dir}")
    all_data = pd.concat(frames, ignore_index=True)

    files_per_day = all_data.groupby("date")["source_file"].nunique()
    if (files_per_day > 1).any():
        problems.append("Some dates appear in more than one export (double counting): "
                        + ", ".join(str(d.date()) for d in files_per_day[files_per_day > 1].index[:5]))

    # Day-level flags
    all_data["is_event_row"] = all_data["category"].isin(EVENT_CATEGORIES)
    event_dates = all_data.loc[all_data["is_event_row"], "date"].unique()
    all_data["is_event_day"] = all_data["date"].isin(event_dates)
    daily_rev = all_data.groupby("date")["net_revenue"].sum()
    all_data["is_open_day"] = all_data["date"].isin(daily_rev[daily_rev > 0].index)

    # Clean names
    all_data["key"] = all_data["product"].map(norm)
    to_classify = all_data["category"].isin(CLASSIFY_CATS)
    all_data.loc[to_classify, "key"] = all_data.loc[to_classify, "key"].str.replace(r"^12\s+", "", regex=True)
    all_data["key"] = all_data["key"].replace(ALIASES)

    # Manual overrides
    all_data.loc[all_data["key"].isin(WINE_OVERRIDE), "group"] = "wine"
    all_data.loc[all_data["key"].isin(["1인 결제완료", "1인 추가결제필요"]), "group"] = "admin"
    all_data.loc[all_data["key"] == "someday", "group"] = "wine"
    all_data.loc[all_data["key"].isin(FOOD_FIX), "group"] = "food"
    all_data.loc[all_data["key"] == "water", "group"] = "drink"
    all_data["is_addon"] = all_data["product"].str.strip().str.startswith("+")
    all_data.loc[all_data["is_addon"], "group"] = "addon"

    # Unlabeled rows: look up names seen in normal categories
    known = all_data[all_data["group"].isin(["food", "wine"])
                     & ~all_data["category"].isin(CLASSIFY_CATS + EVENT_CATEGORIES)]
    lookup = known.groupby("key")["group"].agg(lambda s: s.mode().iloc[0]) if len(known) else pd.Series(dtype=object)
    to_label = to_classify & (all_data["group"] == "other")
    all_data.loc[to_label, "group"] = all_data.loc[to_label, "key"].map(lookup).fillna("unclassified")

    # Price rule + propagation
    is_bottle = (all_data["group"] == "unclassified") & (all_data["unit_price"] >= BOTTLE_PRICE)
    all_data.loc[is_bottle, "group"] = "wine"
    wine_keys = set(all_data.loc[all_data["group"] == "wine", "key"])
    food_keys = set(all_data.loc[all_data["group"] == "food", "key"])
    still = all_data["group"] == "unclassified"
    all_data.loc[still & all_data["key"].isin(wine_keys - food_keys), "group"] = "wine"
    zero = (all_data["group"] == "unclassified") & (all_data["net_revenue"].abs() < 10)
    all_data.loc[zero, "group"] = "zero_price"
    return all_data, problems


def daily_food_table(all_data):
    normal = all_data[~all_data["is_event_day"] & all_data["is_open_day"]]
    food = normal[normal["group"] == "food"]
    daily = food.groupby("date")["qty"].sum().reset_index()
    daily["weekday"] = daily["date"].dt.day_name()
    daily["doy"] = daily["date"].dt.dayofyear
    return normal, daily


def seasonal_pred(date, history, window=21):
    """Mean food count on the same weekday within +-window days of the same time of year."""
    h = history[history["weekday"] == date.day_name()]
    if h.empty:
        return float("nan")
    diff = (h["doy"] - date.dayofyear).abs()
    diff = np.minimum(diff, 365 - diff)
    s = h.loc[diff <= window, "qty"]
    return s.mean() if len(s) >= 3 else h["qty"].mean()


def error_ratios(old_daily):
    """Weekly actual/forecast ratios for 2025, forecasting from 2023-24 only."""
    hist = old_daily[old_daily["date"] < "2025-01-01"]
    test = old_daily[(old_daily["date"] >= "2025-01-01") & old_daily["weekday"].isin(THU_SUN)].copy()
    if hist.empty or test.empty:
        return pd.Series(dtype=float)
    test["pred"] = test["date"].map(lambda d: seasonal_pred(d, hist))
    test["week"] = test["date"].dt.to_period("W")
    wk = test.groupby("week").agg(qty=("qty", "sum"), pred=("pred", "sum"), n=("qty", "size"))
    wk = wk[wk["n"] == 4]
    return wk["qty"] / wk["pred"]


def level_trend(all_data):
    """Weekly actual vs. old-location seasonal baseline at the new location."""
    normal, daily = daily_food_table(all_data)
    old_daily = daily[daily["date"] < OLD_END]
    new_daily = daily[(daily["date"] >= NEW_LOCATION_START) & daily["weekday"].isin(THU_SUN)].copy()
    if new_daily.empty or old_daily.empty:
        return pd.DataFrame()
    new_daily["seasonal"] = new_daily["date"].map(lambda d: seasonal_pred(d, old_daily))
    new_daily["week"] = new_daily["date"].dt.to_period("W")
    t = new_daily.groupby("week").agg(actual=("qty", "sum"), baseline=("seasonal", "sum"), days=("qty", "size"))
    t["ratio"] = (t["actual"] / t["baseline"]).round(2)
    t.index = t.index.astype(str)
    return t


def make_forecast(all_data, planned_days, service_level=0.8, event_extra=0, level_weeks=4):
    """Expected food items and order-up-to quantities for the planned open days."""
    normal, daily = daily_food_table(all_data)
    old_daily = daily[daily["date"] < OLD_END]
    new_daily = daily[(daily["date"] >= NEW_LOCATION_START) & daily["weekday"].isin(THU_SUN)]

    # Level adjustment: recent new-location sales vs. what the old-location seasonal baseline predicted
    level, level_days = 1.0, 0
    if len(new_daily) >= 3 and not old_daily.empty:
        recent = new_daily[new_daily["date"] > new_daily["date"].max() - pd.Timedelta(weeks=level_weeks)]
        if len(recent) >= 3:
            base = recent["date"].map(lambda d: seasonal_pred(d, old_daily)).sum()
            level, level_days = recent["qty"].sum() / base, len(recent)

    # Buffer from past forecast errors
    ratio = error_ratios(old_daily)
    if len(ratio) >= 8:
        mult, buffer_note = float(ratio.quantile(service_level)), f"{len(ratio)} weeks of past errors"
    else:
        mult, buffer_note = 1.2, "default (not enough history)"

    base_total = sum(seasonal_pred(d, old_daily) for d in planned_days)
    expected = base_total * level + event_extra
    order_up_to = base_total * level * mult + event_extra

    # Dish shares: new location if enough days, otherwise same season in earlier years
    food_all = normal[(normal["group"] == "food") & normal["date"].dt.day_name().isin(THU_SUN)]
    new_part = food_all[food_all["date"] >= NEW_LOCATION_START]
    if new_part["date"].nunique() >= 3:
        src, source = new_part, "new location menu"
    else:
        first = min(planned_days)
        dd = (food_all["date"].dt.dayofyear - first.dayofyear).abs()
        dd = np.minimum(dd, 365 - dd)
        src, source = food_all[(dd <= 28) & (food_all["date"] < OLD_END)], "same season, earlier years"
    share = src.groupby("key")["qty"].sum()
    share = (share / share.sum()).sort_values(ascending=False)
    share = share[share.cumsum() - share < 0.95]
    dishes = pd.DataFrame({
        "share": share.round(3),
        "expected": (base_total * level * share).round(1),
        "order_up_to": np.ceil(base_total * level * mult * share).astype(int),
    })
    return {"expected": expected, "order_up_to": order_up_to, "level": level, "level_days": level_days,
            "mult": mult, "buffer_note": buffer_note, "dishes": dishes, "dish_source": source,
            "event_extra": event_extra}