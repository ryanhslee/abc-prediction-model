import pandas as pd

GROUP_MAP = {
    # old location
    "plate": "food",
    "pot": "wine",
    "white": "wine",
    "red": "wine",
    "orange/rose": "wine",
    "pet nat/cremant": "wine",
    # new location
    "🍞 New menu": "food",
    "🍷New pot": "wine",
    "식료품 냉장": "ingredient",
    "식료품 냉동": "ingredient",
    "식료품 드라이": "ingredient",
    "미분류": "other",
}

def load_payhere(path):
    raw = pd.read_excel(path, header=6)
    cat = raw["카테고리명"]

    is_daily_total = cat.str.startswith("일자별 합계", na=False)
    is_total = (cat == "전체 합계") | (cat == "부분 분할결제") | is_daily_total

    products = raw[~is_total].copy()
    products = products.rename(columns={
        "카테고리명": "category", "상품명": "product", "상품별 단가": "unit_price",
        "수량": "qty", "실매출": "net_revenue",
    })
    products["category"] = products["category"].str.split().str.join(" ")
    products["date"] = pd.to_datetime(products["영업일"])
    products = products[["date", "category", "product", "unit_price", "qty", "net_revenue"]]

    # Proof: our per-day revenue must match the file's own daily totals
    reported = raw[is_daily_total].copy()
    reported["date"] = pd.to_datetime(reported["영업일"])
    reported = reported.set_index("date")["실매출"]
    computed = products.groupby("date")["net_revenue"].sum()
    diff = reported.sub(computed, fill_value=0).abs()
    assert diff.max() == 0, f"Daily totals don't match on: {diff[diff > 0].index.tolist()}"

    products["group"] = products["category"].map(GROUP_MAP).fillna("other")
    unmapped = set(products["category"]) - set(GROUP_MAP)
    if unmapped:
        print("Warning: categories not in GROUP_MAP:", unmapped)

    products = products.reset_index(drop=True)

    return products