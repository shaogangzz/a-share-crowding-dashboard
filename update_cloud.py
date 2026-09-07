import json, datetime, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA = Path("data")
DATA.mkdir(exist_ok=True)
HIST = DATA / "history.csv"
OUT = DATA / "latest.json"

def num(v):
    try:
        if v is None: return np.nan
        s = str(v).replace(",", "").replace("%", "").strip()
        return float(s) if s not in ("", "-", "None", "nan") else np.nan
    except Exception:
        return np.nan

def pct_rank_last(series, min_n=20):
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if len(s) < min_n:
        return np.nan
    return float(s.rank(pct=True).iloc[-1] * 100)

def fetch_industries():
    errors = []
    try:
        import akshare as ak
        df = ak.stock_board_industry_spot_em()
        name_col = "板块名称" if "板块名称" in df.columns else "名称"
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "name": str(r[name_col]),
                "close": num(r.get("最新价")),
                "pct": num(r.get("涨跌幅")),
                "turnover": num(r.get("换手率")),
                "amount": num(r.get("成交额")),
                "pe": num(r.get("市盈率-动态")),
                "pb": num(r.get("市净率"))
            })
        if len(rows) >= 20:
            return rows, "AKShare/东方财富"
        errors.append("AKShare returned too few rows")
    except Exception as e:
        errors.append("AKShare: " + repr(e))

    # 直接调用东方财富公开行业行情接口作为备用
    try:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": 1, "pz": 200, "po": 1, "np": 1,
            "fltt": 2, "invt": 2, "fid": "f3",
            "fs": "m:90+t:2+f:!50",
            "fields": "f2,f3,f6,f8,f9,f12,f23"
        }
        j = requests.get(url, params=params, timeout=20,
                         headers={"User-Agent":"Mozilla/5.0"}).json()
        diff = j.get("data", {}).get("diff", [])
        rows = [{
            "name": str(x.get("f12")),
            "close": num(x.get("f2")),
            "pct": num(x.get("f3")),
            "amount": num(x.get("f6")),
            "turnover": num(x.get("f8")),
            "pe": num(x.get("f9")),
            "pb": num(x.get("f23"))
        } for x in diff if x.get("f12")]
        if len(rows) >= 20:
            return rows, "东方财富公开接口"
        errors.append("Eastmoney returned too few rows")
    except Exception as e:
        errors.append("Eastmoney: " + repr(e))
    raise RuntimeError(" | ".join(errors))

today_str = datetime.date.today().isoformat()
try:
    raw, source = fetch_industries()
    today = pd.DataFrame(raw)
    today.insert(0, "date", today_str)

    if HIST.exists():
        hist = pd.read_csv(HIST)
    else:
        hist = pd.DataFrame(columns=today.columns)
    hist = hist[hist["date"].astype(str) != today_str] if len(hist) else hist
    hist = pd.concat([hist, today], ignore_index=True)
    hist.to_csv(HIST, index=False)

    # 首日/历史不足时使用当日行业横截面分位数，避免空白；历史积累后切换为750日滚动分位。
    cross = today.set_index("name")
    cross_price = cross["pct"].rank(pct=True) * 100
    cross_volume = cross["amount"].rank(pct=True) * 100
    cross_turn = cross["turnover"].rank(pct=True) * 100
    cross_flow = cross["amount"].pct_change().rank(pct=True) * 100 if len(cross) else pd.Series(dtype=float)
    cross_val = ((cross["pe"].rank(pct=True) + cross["pb"].rank(pct=True)) / 2) * 100

    out = []
    for name, g in hist.groupby("name"):
        g = g.sort_values("date").tail(750).copy()
        z = g.iloc[-1]
        if len(g) >= 61 and pd.notna(z.get("close")):
            r20 = z["close"] / g["close"].iloc[-21] - 1
            r60 = z["close"] / g["close"].iloc[-61] - 1
            price = np.nanmean([pct_rank_last(g["close"].pct_change(20)), pct_rank_last(g["close"].pct_change(60))])
        else:
            r20 = 0
            price = cross_price.get(name, 50.0)

        volume = pct_rank_last(g["amount"])
        turnover = pct_rank_last(g["turnover"])
        valuation = np.nanmean([pct_rank_last(g["pe"]), pct_rank_last(g["pb"])])
        flow = pct_rank_last(g["amount"].pct_change(5))

        # 历史不足时采用横截面临时评分，保证首日即可展示；随后逐步切换历史分位。
        if pd.isna(volume): volume = cross_volume.get(name, 50.0)
        if pd.isna(turnover): turnover = cross_turn.get(name, 50.0)
        if pd.isna(flow): flow = cross_price.get(name, 50.0)
        if pd.isna(valuation): valuation = cross_val.get(name, 50.0)
        if pd.isna(price): price = 50.0

        score = 0.25*price + 0.20*volume + 0.20*turnover + 0.20*flow + 0.15*valuation
        score = float(np.clip(score, 0, 100))
        opportunity = float(np.clip(0.55*(100-score) + 0.25*price + 0.20*(100-valuation), 0, 100))

        risk = "极度拥挤" if score >= 85 else "高拥挤" if score >= 70 else "中度拥挤" if score >= 55 else "中性" if score >= 40 else "低拥挤"
        out.append({
            "name": name, "crowding": round(score,1),
            "change20": round(float(r20)*100,1) if pd.notna(r20) else 0.0,
            "risk": risk, "opportunity": round(opportunity,1)
        })

    out.sort(key=lambda x: x["crowding"], reverse=True)
    high = [x["name"] for x in out[:3]]
    opp = sorted(out, key=lambda x: x["opportunity"], reverse=True)[:3]
    quality = ("历史数据初始化阶段（当前采用行业横截面分位，历史积累后自动切换滚动分位）"
               if len(hist["date"].unique()) < 20 else "历史滚动分位模型")
    summary = "当前拥挤度最高的行业包括：" + "、".join(high) + "。机会雷达优先关注：" + "、".join(x["name"] for x in opp) + "。高拥挤意味着交易一致性较高，应重点防范波动放大和资金反转。"
    result = {
        "date": today_str,
        "status": "云端自动更新成功",
        "data_quality": quality + "｜数据源：" + source,
        "summary": summary,
        "industries": out
    }
except Exception as e:
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"industries":[]}
    result = old
    result["date"] = today_str
    result["status"] = "数据更新失败：" + repr(e)[:180]

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(result["status"])
print("industries:", len(result.get("industries", [])))
