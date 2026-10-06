from pathlib import Path
import re
import pandas as pd
from parser import load_payhere

DATA = Path(__file__).parent / "data"

# ---------- Settings ----------
EVENT_CATEGORIES = [
    "abc market 🧺", "janemarch mkt", "25.04.26 제인마치",
    "sunday heaven", "🦪3.10 sunday🌿",
    "pot 3주년", "🌷jenny 시음회", "viaplain🧡💛", "타르틴", "🌿green j", "⚽️니벨크랙🍷",
    "10/11(대관)", "daycare center", "wendys 5yrs",
]
SEASONAL = ["❄️12월", "❄️12월 pot", "🎄", "🔴 red abc"]
WINE_OVERRIDE = ["daily glass", "half bottle", "half", "tmb half", "chillable red", "acid frick", "carbonic maceration"]
BOTTLE_PRICE = 80000   # unclassified rows at or above this price are treated as wine

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

is_seasonal = all_data["category"].isin(SEASONAL)
all_data.loc[is_seasonal, "key"] = (
    all_data.loc[is_seasonal, "key"].str.replace(r"^12\s+", "", regex=True)
)

# ---------- 4. Manual overrides ----------
all_data.loc[all_data["key"].isin(WINE_OVERRIDE), "group"] = "wine"
all_data.loc[all_data["key"] == "1인 결제완료", "group"] = "admin"

all_data["is_addon"] = all_data["product"].str.strip().str.startswith("+")
all_data.loc[all_data["is_addon"], "group"] = "addon"

# ---------- 5. Seasonal rows: look up each name in normal categories ----------
known = all_data[all_data["group"].isin(["food", "wine"])
                 & ~all_data["category"].isin(SEASONAL + EVENT_CATEGORIES)]
n_groups = known.groupby("key")["group"].nunique()
print("names found under both food and wine:", n_groups[n_groups > 1].index.tolist())
lookup = known.groupby("key")["group"].agg(lambda s: s.mode().iloc[0])

to_label = is_seasonal & (all_data["group"] == "other")
all_data.loc[to_label, "group"] = all_data.loc[to_label, "key"].map(lookup).fillna("unclassified")

# ---------- 6. Price rule for what is still unclassified ----------
is_bottle = (all_data["group"] == "unclassified") & (all_data["unit_price"] >= BOTTLE_PRICE)
all_data.loc[is_bottle, "group"] = "wine"

# Propagate: a name already labeled wine (e.g., via the bottle price) is wine at any price
wine_keys = set(all_data.loc[all_data["group"] == "wine", "key"])
food_keys = set(all_data.loc[all_data["group"] == "food", "key"])
still = all_data["group"] == "unclassified"
all_data.loc[still & all_data["key"].isin(wine_keys - food_keys), "group"] = "wine"

# Small manual labels
FOOD_FIX = ["pappardelle", "cheese olives", "spicy pepperoni", "charcueterie"]
all_data.loc[still & all_data["key"].isin(FOOD_FIX), "group"] = "food"
all_data.loc[all_data["key"] == "1인 추가결제필요", "group"] = "admin"
all_data.loc[all_data["key"] == "water", "group"] = "other"

# Zero-revenue leftovers (modifier/note lines)
zero = (all_data["group"] == "unclassified") & (all_data["net_revenue"].abs() < 10)
all_data.loc[zero, "group"] = "zero_price"

# ---------- 7. Checks ----------
print("event days:", len(event_dates))
print("still unclassified:")
print(all_data[all_data["group"] == "unclassified"]
      .groupby("key")[["qty", "net_revenue"]].sum()
      .sort_values("net_revenue", ascending=False).head(15))

# ---------- 8. Ingredient log vs. food, by quarter (normal open days) ----------
normal = all_data[~all_data["is_event_day"] & all_data["is_open_day"]]
cov = normal.pivot_table(index="source_file", columns="group", values="qty", aggfunc="sum")
cov["days_open"] = normal.groupby("source_file")["date"].nunique()
cov["days_with_log"] = (normal[normal["group"] == "ingredient"]
                        .groupby("source_file")["date"].nunique())
cov["food_per_day"] = cov["food"] / cov["days_open"]
cov["log_per_day"] = cov["ingredient"] / cov["days_open"]
cov["log_per_food"] = cov["ingredient"] / cov["food"]
print(cov.round(2).to_string())

# ---------- 9. Before/after Oct 2024 (normal open days) ----------
win = normal[(normal["date"] >= "2023-07-01") & (normal["date"] <= "2025-12-31")].copy()
win["after"] = win["date"] >= "2024-10-01"
open_days = win.groupby("after")["date"].nunique()

def per_day(group):
    t = win[win["group"] == group].groupby(["after", "key"])["qty"].sum().unstack("after").fillna(0)
    t = t / open_days
    t["change"] = t[True] - t[False]
    return t.sort_values("change")

print(per_day("ingredient").head(10))
print(per_day("food").tail(10))


#--------------Demand Patterns------------
food = normal[normal["group"] == "food"]
daily = food.groupby("date")["qty"].sum().reset_index()
daily["weekday"] = daily["date"].dt.day_name()
daily["month"] = daily["date"].dt.month

print(daily.groupby("weekday")["qty"].agg(["mean", "count"]).round(1))
print(daily.groupby("month")["qty"].agg(["mean", "count"]).round(1))

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily["weekday"] = pd.Categorical(daily["weekday"], categories=order, ordered=True)
daily["year"] = daily["date"].dt.year
print(daily.pivot_table(index="weekday", columns="year", values="qty", aggfunc=["mean", "count"], observed=False).round(1))

##----baseline forecast----
# daily = daily.sort_values("date").reset_index(drop=True)
# g = daily.groupby("weekday", observed=True)["qty"]
# daily["pred_last"] = g.shift(1)
# daily["pred_avg4"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())

# test = daily[daily["date"] >= "2024-01-01"].dropna(subset=["pred_last", "pred_avg4"])

# def wape(actual, pred):
#     return (actual - pred).abs().sum() / actual.sum()

# def bias(actual, pred):
#     return (pred - actual).sum() / actual.sum()

# for name in ["pred_last", "pred_avg4"]:
#     print(name, "WAPE:", round(wape(test["qty"], test[name]), 3),
#           "bias:", round(bias(test["qty"], test[name]), 3))

daily = daily.sort_values("date").reset_index(drop=True)
g = daily.groupby("weekday", observed=True)["qty"]
daily["pred_last"] = g.shift(1)
for k in [4, 8, 12]:
    daily[f"pred_avg{k}"] = g.transform(lambda s, k=k: s.shift(1).rolling(k, min_periods=2).mean())

preds = ["pred_last", "pred_avg4", "pred_avg8", "pred_avg12"]
test = daily[daily["date"] >= "2024-01-01"].dropna(subset=preds).copy()
test["week"] = test["date"].dt.to_period("W")
weekly = test.groupby("week")[["qty"] + preds].sum()

def wape(actual, pred):
    return (actual - pred).abs().sum() / actual.sum()

for name in preds:
    print(name, "daily WAPE:", round(wape(test["qty"], test[name]), 3),
          "| weekly WAPE:", round(wape(weekly["qty"], weekly[name]), 3))

test["abs_err"] = (test["qty"] - test["pred_avg4"]).abs()

by_day = test.groupby("weekday", observed=True)[["abs_err", "qty"]].sum()
by_day["wape"] = by_day["abs_err"] / by_day["qty"]
print(by_day.round(3))

big = test.nlargest(15, "abs_err").copy()
big["err"] = big["qty"] - big["pred_avg4"]
print(big[["date", "weekday", "qty", "pred_avg4", "err"]]) 

dates = big["date"]
sub = all_data[all_data["date"].isin(dates)]

chk = sub.pivot_table(index="date", columns="group", values="qty", aggfunc="sum", fill_value=0)
chk["revenue"] = sub.groupby("date")["net_revenue"].sum()

STANDARD = ["plate", "pot", "white", "red", "orange/rose", "pet nat/cremant",
            "식료품 냉장", "식료품 냉동", "식료품 드라이"]
chk["other_categories"] = (sub[~sub["category"].isin(STANDARD)]
                           .groupby("date")["category"].agg(lambda s: ", ".join(sorted(set(s)))))
print(chk.to_string())