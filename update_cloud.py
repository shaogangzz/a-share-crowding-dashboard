import json, datetime, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA=Path("data"); DATA.mkdir(exist_ok=True)
HIST=DATA/"history.csv"; OUT=DATA/"latest.json"

def num(v):
    try:
        if v is None: return np.nan
        s=str(v).replace(",","").replace("%","").strip()
        x=float(s)
        return x if math.isfinite(x) else np.nan
    except: return np.nan

def clean(v, default=50.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except: return default

def rank_now(s, name, default=50.0):
    x=pd.to_numeric(s, errors="coerce").dropna()
    if len(x)<2 or name not in s.index: return default
    v=s.rank(pct=True).get(name)
    return clean(v*100 if pd.notna(v) else default, default)

def hist_rank(s, min_n=20):
    x=pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if len(x)<min_n: return np.nan
    return clean(x.rank(pct=True).iloc[-1]*100, np.nan)

def fetch_rows():
    errors=[]
    try:
        import akshare as ak
        df=ak.stock_board_industry_spot_em()
        nc="板块名称" if "板块名称" in df.columns else "名称"
        rows=[{"name":str(r[nc]),"close":num(r.get("最新价")),"pct":num(r.get("涨跌幅")),"amount":num(r.get("成交额")),"turnover":num(r.get("换手率")),"pe":num(r.get("市盈率-动态")),"pb":num(r.get("市净率"))} for _,r in df.iterrows()]
        if len(rows)>=20: return rows,"AKShare/东方财富"
    except Exception as e: errors.append(repr(e))
    try:
        j=requests.get("https://push2.eastmoney.com/api/qt/clist/get",params={"pn":1,"pz":200,"po":1,"np":1,"fltt":2,"invt":2,"fid":"f3","fs":"m:90+t:2+f:!50","fields":"f2,f3,f6,f8,f9,f12,f14,f23"},headers={"User-Agent":"Mozilla/5.0"},timeout=20).json()
        rows=[]
        for x in j.get("data",{}).get("diff",[]):
            rows.append({"name":str(x.get("f14") or x.get("f12")),"close":num(x.get("f2")),"pct":num(x.get("f3")),"amount":num(x.get("f6")),"turnover":num(x.get("f8")),"pe":num(x.get("f9")),"pb":num(x.get("f23"))})
        if len(rows)>=20: return rows,"东方财富公开接口"
    except Exception as e: errors.append(repr(e))
    raise RuntimeError(" | ".join(errors))

today_date=datetime.date.today().isoformat()
try:
    rows,source=fetch_rows()
    today=pd.DataFrame(rows)
    today.insert(0,"date",today_date)
    hist=pd.read_csv(HIST) if HIST.exists() else pd.DataFrame(columns=today.columns)
    hist=hist[hist["date"].astype(str)!=today_date] if len(hist) else hist
    hist=pd.concat([hist,today],ignore_index=True)
    hist.to_csv(HIST,index=False)

    c=today.set_index("name")
    out=[]
    for name,g in hist.groupby("name"):
        g=g.sort_values("date").tail(750)
        z=g.iloc[-1]
        if name not in c.index: continue
        price=hist_rank(g["close"].pct_change(20))
        if not math.isfinite(clean(price,np.nan)): price=rank_now(c["pct"],name)
        volume=hist_rank(g["amount"])
        if not math.isfinite(clean(volume,np.nan)): volume=rank_now(c["amount"],name)
        turnover=hist_rank(g["turnover"])
        if not math.isfinite(clean(turnover,np.nan)): turnover=rank_now(c["turnover"],name)
        flow=hist_rank(g["amount"].pct_change(5))
        if not math.isfinite(clean(flow,np.nan)): flow=rank_now(c["pct"],name)
        pe=hist_rank(g["pe"]); pb=hist_rank(g["pb"])
        valuation=np.nanmean([pe,pb])
        if not math.isfinite(clean(valuation,np.nan)):
            rv1=rank_now(c["pe"],name); rv2=rank_now(c["pb"],name)
            valuation=clean(np.nanmean([rv1,rv2]),50)

        price,volume,turnover,flow,valuation=[clean(v,50) for v in (price,volume,turnover,flow,valuation)]
        score=round(float(np.clip(.25*price+.20*volume+.20*turnover+.20*flow+.15*valuation,0,100)),1)
        opp=round(float(np.clip(.55*(100-score)+.25*price+.20*(100-valuation),0,100)),1)
        risk="极度拥挤" if score>=85 else "高拥挤" if score>=70 else "中度拥挤" if score>=55 else "中性" if score>=40 else "低拥挤"
        change20=0.0
        if len(g)>=21:
            a,b=num(z.get("close")),num(g.iloc[-21].get("close"))
            if pd.notna(a) and pd.notna(b) and b!=0: change20=round((a/b-1)*100,1)
        out.append({"name":name,"crowding":score,"change20":change20,"risk":risk,"opportunity":opp})

    out.sort(key=lambda x:x["crowding"],reverse=True)
    top=[x["name"] for x in out[:3]]
    opps=sorted(out,key=lambda x:x["opportunity"],reverse=True)[:3]
    days=hist["date"].nunique()
    quality=("初始化阶段：行业横截面分位 + 已积累 "+str(days)+" 个交易日" if days<20 else "750日历史滚动分位模型")
    result={"date":today_date,"status":"云端自动更新成功","data_quality":quality+"｜数据源："+source,"summary":"当前拥挤度较高："+ "、".join(top)+"。机会雷达关注："+ "、".join(x["name"] for x in opps)+"。","industries":out}
except Exception as e:
    result={"date":today_date,"status":"数据更新失败："+repr(e)[:180],"data_quality":"--","summary":"本次数据更新失败，保留系统等待下一次更新。","industries":[]}

OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
print(result["status"],"industries:",len(result["industries"]))
