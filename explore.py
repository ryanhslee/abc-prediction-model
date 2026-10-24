from pathlib import Path
import re
import pandas as pd
from parser import load_payhere

DATA = Path(__file__).parent / "data"

# ---------- Settings ----------
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
BOTTLE_PRICE = 80000
THU_SUN = ["Thursday", "Friday", "Saturday", "Sunday"]

# ---------- 1. Load all files ----------
frames = []
for f in sorted(DATA.glob("*.xlsx")):
    try:
        df = load_payhere(f)
    except AssertionError as e:
        print("FAILED:", f.name, e)
        continue
    df["source_file"] = f.name
    frames.append(df)
all_data = pd.concat(frames, ignore_index=True)

# ---------- 2. Day-level flags ----------
all_data["is_event_row"] = all_data["category"].isin(EVENT_CATEGORIES)
event_dates = all_data.loc[all_data["is_event_row"], "date"].unique()
all_data["is_event_day"] = all_data["date"].isin(event_dates)

daily_rev = all_data.groupby("date")["net_revenue"].sum()
all_data["is_open_day"] = all_data["date"].isin(daily_rev[daily_rev > 0].index)

# ---------- 3. Clean product names ----------
def norm(name):
    s = str(name).lower().strip()
    s = re.sub(r"^[>+\s]+", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

all_data["key"] = all_data["product"].map(norm)
to_classify = all_data["category"].isin(CLASSIFY_CATS)
all_data.loc[to_classify, "key"] = (
    all_data.loc[to_classify, "key"].str.replace(r"^12\s+", "", regex=True)
)
ALIASES = {"chicken wig": "chicken wing"}
all_data["key"] = all_data["key"].replace(ALIASES)

# ---------- 4. Manual overrides ----------
all_data.loc[all_data["key"].isin(WINE_OVERRIDE), "group"] = "wine"
all_data.loc[all_data["key"].isin(["1인 결제완료", "1인 추가결제필요"]), "group"] = "admin"
all_data.loc[all_data["key"] == "someday", "group"] = "wine"
all_data.loc[all_data["key"].isin(FOOD_FIX), "group"] = "food"
all_data.loc[all_data["key"] == "water", "group"] = "drink"

all_data["is_addon"] = all_data["product"].str.strip().str.startswith("+")
all_data.loc[all_data["is_addon"], "group"] = "addon"

# ---------- 5. Unlabeled rows: look up names seen in normal categories ----------
known = all_data[all_data["group"].isin(["food", "wine"])
                 & ~all_data["category"].isin(CLASSIFY_CATS + EVENT_CATEGORIES)]
lookup = known.groupby("key")["group"].agg(lambda s: s.mode().iloc[0])
to_label = to_classify & (all_data["group"] == "other")
all_data.loc[to_label, "group"] = all_data.loc[to_label, "key"].map(lookup).fillna("unclassified")

# ---------- 6. Price rule + propagation for what is still unclassified ----------
is_bottle = (all_data["group"] == "unclassified") & (all_data["unit_price"] >= BOTTLE_PRICE)
all_data.loc[is_bottle, "group"] = "wine"

wine_keys = set(all_data.loc[all_data["group"] == "wine", "key"])
food_keys = set(all_data.loc[all_data["group"] == "food", "key"])
still = all_data["group"] == "unclassified"
all_data.loc[still & all_data["key"].isin(wine_keys - food_keys), "group"] = "wine"

zero = (all_data["group"] == "unclassified") & (all_data["net_revenue"].abs() < 10)
all_data.loc[zero, "group"] = "zero_price"

# ---------- 7. Checks ----------
STANDARD = ["plate", "pot", "white", "red", "orange/rose", "pet nat/cremant",
            "식료품 냉장", "식료품 냉동", "식료품 드라이", "🍞 New menu", "🍷New pot"]
print("1) event days:", len(event_dates))
print("2) categories not yet handled:",
      sorted(set(all_data["category"]) - set(STANDARD) - set(EVENT_CATEGORIES) - set(CLASSIFY_CATS)))
print("3) still unclassified:")
print(all_data[all_data["group"] == "unclassified"]
      .groupby("key")[["qty", "net_revenue"]].sum()
      .sort_values("net_revenue", ascending=False).head(10))

# ---------- 8. Baseline: Thu-Sun food items on normal open days (old location) ----------
def wape(actual, pred):
    return (actual - pred).abs().sum() / actual.sum()

normal = all_data[~all_data["is_event_day"] & all_data["is_open_day"]]
food = normal[(normal["group"] == "food") & (normal["date"] < "2026-01-01")]
daily = food.groupby("date")["qty"].sum().reset_index()
daily["weekday"] = daily["date"].dt.day_name()

core = daily[daily["weekday"].isin(THU_SUN)].sort_values("date").reset_index(drop=True)
g = core.groupby("weekday")["qty"]
core["pred_last"] = g.shift(1)
core["pred_avg4"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())

test = core[core["date"] >= "2024-01-01"].dropna(subset=["pred_last", "pred_avg4"]).copy()
test["week"] = test["date"].dt.to_period("W")
n_days = test.groupby("week")["qty"].size()
weekly = test.groupby("week")[["qty", "pred_last", "pred_avg4"]].sum()
weekly = weekly[n_days == 4]

print("4) Thu-Sun baselines")
for name in ["pred_last", "pred_avg4"]:
    print(name, "daily WAPE:", round(wape(test["qty"], test[name]), 3),
          "| weekly WAPE:", round(wape(weekly["qty"], weekly[name]), 3))
print("5) days scored:", len(test), "| complete weeks scored:", len(weekly))

# ---------- 9. Model: Poisson regression with calendar features, weekly rolling backtest ----------
import numpy as np
import holidays
from sklearn.linear_model import PoissonRegressor

def bias(actual, pred):
    return (pred - actual).sum() / actual.sum()

kr = holidays.KR(years=range(2023, 2027))
frame = core.dropna(subset=["pred_avg4"]).copy().reset_index(drop=True)
frame["is_holiday"] = frame["date"].dt.date.map(lambda d: d in kr).astype(float)
frame["pre_holiday"] = frame["date"].map(lambda d: (d + pd.Timedelta(days=1)).date() in kr).astype(float)
frame["log_avg4"] = np.log1p(frame["pred_avg4"])
frame["week_start"] = frame["date"].dt.to_period("W").dt.start_time

wd_cols = []
for d in THU_SUN:
    frame["wd_" + d] = (frame["weekday"] == d).astype(float)
    wd_cols.append("wd_" + d)
mo_cols = []
for mth in range(1, 13):
    frame["mo_%d" % mth] = (frame["date"].dt.month == mth).astype(float)
    mo_cols.append("mo_%d" % mth)
hol_cols = ["is_holiday", "pre_holiday"]
lag_cols = ["log_avg4"]

weeks = sorted(frame.loc[frame["date"] >= "2024-01-01", "week_start"].unique())

def backtest(cols, alpha=1.0, window_weeks=None):
    out = []
    for ws in weeks:
        train = frame[frame["date"] < ws]
        if window_weeks:
            train = train[train["date"] >= ws - pd.Timedelta(weeks=window_weeks)]
        tw = frame[frame["week_start"] == ws]
        if len(tw) == 0 or len(train) < 60:
            continue
        m = PoissonRegressor(alpha=alpha, max_iter=3000).fit(train[cols], train["qty"])
        out.append(pd.Series(m.predict(tw[cols]), index=tw.index))
    return pd.concat(out)

def score(pred):
    ev = frame.loc[pred.index, ["qty", "week_start"]].copy()
    ev["pred"] = pred
    wk = ev.groupby("week_start").agg(qty=("qty", "sum"), pred=("pred", "sum"), n=("qty", "size"))
    wk = wk[wk["n"] == 4]
    return (round(wape(ev["qty"], ev["pred"]), 3), round(wape(wk["qty"], wk["pred"]), 3),
            round(bias(ev["qty"], ev["pred"]), 3))

feature_sets = {
    "weekday only": wd_cols,
    "weekday + avg4": wd_cols + lag_cols,
    "weekday + month": wd_cols + mo_cols,
    "weekday + month + holiday": wd_cols + mo_cols + hol_cols,
    "weekday + month + holiday + avg4": wd_cols + mo_cols + hol_cols + lag_cols,
}

results, preds = [], {}
for name, cols in feature_sets.items():
    preds[name] = backtest(cols)
idx = preds["weekday only"].index
results.append(("baseline: avg of last 4 same weekdays",) + score(frame.loc[idx, "pred_avg4"]))
for name in feature_sets:
    results.append((name,) + score(preds[name]))

print("6) Model comparison (same days for every row)")
print(pd.DataFrame(results, columns=["model", "daily WAPE", "weekly WAPE", "bias"]).to_string(index=False))
print("days scored:", len(idx))

print("7) Training-window test")
best_sets = ["weekday + avg4", "weekday + month + holiday + avg4"]
runs = {}
for name in best_sets:
    for w in [None, 52, 26]:
        runs[(name, w)] = backtest(feature_sets[name], window_weeks=w)
common = idx
for p in runs.values():
    common = common.intersection(p.index)
rows = [("baseline",) + score(frame.loc[common, "pred_avg4"])]
for (name, w), p in runs.items():
    rows.append((name + " | train on " + str(w or "all") + " weeks",) + score(p.loc[common]))
print(pd.DataFrame(rows, columns=["model", "daily WAPE", "weekly WAPE", "bias"]).to_string(index=False))
print("days scored:", len(common))

# ---------- 10. Weekly order suggestion (v1, dish level, seasonal baseline) ----------
SERVICE_LEVEL = 0.80   # chance of NOT running out; see newsvendor note
NEW_LOCATION_START = "2026-09-01"
PLANNED_DAYS = pd.to_datetime(["2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01"])  # change weekly

old_daily = daily[daily["date"] < "2026-01-01"].copy()
old_daily["doy"] = old_daily["date"].dt.dayofyear

def seasonal_pred(date, history=None, window=21):
    h = old_daily if history is None else history
    h = h[h["weekday"] == date.day_name()]
    diff = (h["doy"] - date.dayofyear).abs()
    diff = np.minimum(diff, 365 - diff)
    s = h.loc[diff <= window, "qty"]
    return s.mean() if len(s) >= 3 else h["qty"].mean()

# Check: predict 2025 using only 2023-24 history, vs. the 4-week average, same days
hist = old_daily[old_daily["date"] < "2025-01-01"]
chk = old_daily[(old_daily["date"] >= "2025-01-01") & old_daily["weekday"].isin(THU_SUN)].copy()
chk["seasonal"] = chk["date"].map(lambda d: seasonal_pred(d, history=hist))
chk = chk.merge(core[["date", "pred_avg4"]], on="date", how="left").dropna(subset=["seasonal", "pred_avg4"])
chk["week"] = chk["date"].dt.to_period("W")
wk = chk.groupby("week").agg(qty=("qty", "sum"), seasonal=("seasonal", "sum"),
                             avg4=("pred_avg4", "sum"), n=("qty", "size"))
wk = wk[wk["n"] == 4]
print("7) 2025 check, seasonal (trained on 2023-24 only) vs avg4 | daily:",
      round(wape(chk["qty"], chk["seasonal"]), 3), "vs", round(wape(chk["qty"], chk["pred_avg4"]), 3),
      "| weekly:", round(wape(wk["qty"], wk["seasonal"]), 3), "vs", round(wape(wk["qty"], wk["avg4"]), 3),
      "| weeks:", len(wk))

# Level adjustment for the new location
new_food = normal[(normal["group"] == "food") & (normal["date"] >= NEW_LOCATION_START)]
new_daily = new_food.groupby("date")["qty"].sum().reset_index()
new_daily["weekday"] = new_daily["date"].dt.day_name()
new_daily = new_daily[new_daily["weekday"].isin(THU_SUN)]
if len(new_daily) >= 3:
    level_adjust = new_daily["qty"].sum() / new_daily["date"].map(seasonal_pred).sum()
else:
    level_adjust = 1.0
print("8) new-location Thu-Sun days used:", len(new_daily), "| level adjustment:", round(level_adjust, 2))
nd = new_daily.copy()
nd["seasonal"] = nd["date"].map(seasonal_pred)
nd["week"] = nd["date"].dt.to_period("W")
trend = nd.groupby("week").agg(actual=("qty", "sum"), seasonal=("seasonal", "sum"), days=("qty", "size"))
trend["ratio"] = (trend["actual"] / trend["seasonal"]).round(2)
print("   weekly level ratio at new location:")
print(trend.to_string())

# Forecast and buffer
expected_total = sum(seasonal_pred(d) for d in PLANNED_DAYS) * level_adjust
ratio = wk["qty"] / wk["seasonal"]
print("   weekly actual/seasonal ratio quantiles:", ratio.quantile([0.1, 0.5, 0.8, 0.9]).round(2).to_dict(),
      "| weeks:", len(ratio))
mult = ratio.quantile(SERVICE_LEVEL)
order_up_to = expected_total * mult
print("   expected food items:", round(expected_total), "| order up to:", round(order_up_to),
      "| buffer multiplier:", round(mult, 2))

# Dish shares: new menu if there is enough new-location data, else the same season in earlier years
food_all = normal[(normal["group"] == "food") & normal["date"].dt.day_name().isin(THU_SUN)]
new_part = food_all[food_all["date"] >= NEW_LOCATION_START]
if new_part["date"].nunique() >= 3:
    src, label = new_part, "new location"
else:
    dd = (food_all["date"].dt.dayofyear - PLANNED_DAYS[0].dayofyear).abs()
    dd = np.minimum(dd, 365 - dd)
    src, label = food_all[(dd <= 28) & (food_all["date"] < "2026-01-01")], "same season, earlier years"
share = src.groupby("key")["qty"].sum()
share = (share / share.sum()).sort_values(ascending=False)
share = share[share.cumsum() - share < 0.95]
order = pd.DataFrame({"share": share.round(3),
                      "expected": (expected_total * share).round(1),
                      "order_up_to": np.ceil(order_up_to * share).astype(int)})
print("9) dish quantities from:", label)
print(order.head(15).to_string())