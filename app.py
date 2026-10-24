import hmac
import os
import time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
from pipeline import build_dataset, make_forecast, level_trend

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

st.set_page_config(page_title="주간 발주 도우미", layout="wide")


def require_password():
    """Block the whole app until the correct password is entered."""
    try:
        expected = st.secrets["password"]
    except Exception:
        expected = os.environ.get("APP_PASSWORD")
    if not expected:
        st.error("비밀번호가 설정되지 않았습니다. .streamlit/secrets.toml 파일에 password 를 설정해 주세요.")
        st.stop()
    if st.session_state.get("authed"):
        return
    st.title("주간 발주 도우미")
    with st.form("login"):
        pw = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인")
    if submitted:
        if hmac.compare_digest(pw.encode(), str(expected).encode()):
            st.session_state["authed"] = True
            st.rerun()
        time.sleep(1)
        st.error("비밀번호가 올바르지 않습니다.")
    st.stop()


require_password()
st.title("주간 발주 도우미 (v1)")
st.caption("과거 판매 기록으로 이번 주 음식 판매량을 예측하고, 발주 상한선을 계산합니다.")


@st.cache_data(show_spinner="데이터 불러오는 중...")
def load(stamp):
    return build_dataset(DATA)


# ---------- Sidebar: add new exports ----------
with st.sidebar:
    if st.button("로그아웃"):
        st.session_state["authed"] = False
        st.rerun()
    st.header("데이터")
    files = st.file_uploader("Payhere 엑셀 파일 추가", type="xlsx", accept_multiple_files=True)
    if files and st.button("저장하고 새로고침"):
        for f in files:
            (DATA / f.name).write_bytes(f.getbuffer())
        st.cache_data.clear()
        st.rerun()
    st.caption("같은 기간이 겹치는 파일을 올리면 매출이 이중 집계됩니다.")

stamp = tuple((f.name, f.stat().st_mtime) for f in sorted(DATA.glob("*.xlsx")))
try:
    all_data, problems = load(stamp)
except FileNotFoundError:
    st.error("data 폴더에 엑셀 파일이 없습니다. 왼쪽에서 Payhere 파일을 추가해 주세요.")
    st.stop()

for p in problems:
    st.warning(p)
last_day = all_data.loc[all_data["is_open_day"], "date"].max().date()
st.write(f"데이터 마지막 영업일: **{last_day}**")

# ---------- Inputs ----------
st.subheader("1. 이번 주 영업일")
today = date.today()
first = today + timedelta(days=(3 - today.weekday()) % 7)      # next Thursday (or today)
options = [today + timedelta(days=i) for i in range(0, 15)]
default = [first + timedelta(days=i) for i in range(4) if first + timedelta(days=i) in options]
planned = st.multiselect("영업하는 날짜를 선택하세요", options, default=default,
                         format_func=lambda d: d.strftime("%m/%d (%a)"))

col1, col2 = st.columns(2)
with col1:
    event_extra = st.number_input("행사/단체 예약으로 추가되는 음식 수량 (없으면 0)", 0, 500, 0)
with col2:
    service = st.slider("재고 부족을 피하고 싶은 정도 (높을수록 넉넉하게 발주)", 0.50, 0.95, 0.80, 0.05)

if not planned:
    st.info("영업일을 하나 이상 선택해 주세요.")
    st.stop()

r = make_forecast(all_data, [pd.Timestamp(d) for d in planned], service, event_extra)

# ---------- Results ----------
st.subheader("2. 예상 판매량과 발주 상한")
m1, m2, m3, m4 = st.columns(4)
m1.metric("예상 음식 판매 수량", f"{r['expected']:.0f}")
m2.metric("발주 상한 (이만큼까지)", f"{r['order_up_to']:.0f}")
m3.metric("새 매장 보정 계수", f"{r['level']:.2f}")
m4.metric("여유분 배수", f"{r['mult']:.2f}")
st.caption(f"여유분 기준: {r['buffer_note']}. 보정 계수는 최근 {r['level_days']}일 데이터로 계산"
           + (" (데이터가 부족해 1.0 적용)" if r["level_days"] == 0 else "") + ".")

st.subheader("3. 메뉴별 발주 상한")
dishes = r["dishes"].reset_index().rename(columns={
    "key": "메뉴", "share": "비중", "expected": "예상 수량", "order_up_to": "발주 상한"})
dishes["비중"] = dishes["비중"].map(lambda x: f"{x * 100:.1f}%")
st.dataframe(dishes, hide_index=True)
st.caption(f"메뉴 비중 기준: {r['dish_source']}. 행사 추가분({r['event_extra']})은 메뉴별로 나누지 않았습니다. "
           "메뉴별 수량은 변동이 커서 작은 메뉴는 여유 있게 잡으세요.")

st.subheader("4. 새 매장 판매 추이 (예전 매장 같은 시기 대비)")
trend = level_trend(all_data)
if trend.empty:
    st.write("새 매장 데이터가 아직 충분하지 않습니다.")
else:
    st.line_chart(trend["ratio"])
    st.dataframe(trend)
st.caption("비율이 1이면 예전 매장과 같은 수준입니다. 영업일이 2~3일인 주는 변동이 큽니다.")