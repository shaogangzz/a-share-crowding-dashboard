import json, datetime, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA=Path("data"); DATA.mkdir(exist_ok=True)
HIST=DATA/"history.csv"; OUT=DATA/"latest.json"; SERIES=DATA/"timeseries.json"
UA={"User-Agent":"Mozilla/5.0"}

def num(v):
    try:
        if v is None: return np.nan
        x=float(str(v).replace(",","").replace("%","").strip())
        return x if math.isfinite(x) else np.nan
    except: return np.nan

def finite(v, default=50.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except: return default

def fetch_rows():
    errors=[]
    try:
        import akshare as ak
        df=ak.stock_board_industry_spot_em()
        nc="板块名称" if "板块名称" in df.columns else "名称"
        rows=[{"code":str(r.get("板块代码","")),"name":str(r[nc]),"close":num(r.get("最新价")),"pct":num(r.get("涨跌幅")),"amount":num(r.get("成交额")),"turnover":num(r.get("换手率")),"pe":num(r.get("市盈率-动态")),"pb":num(r.get("市净率"))} for _,r in df.iterrows()]
        if len(rows)>=20:return rows,"AKShare/东方财富"
    except Exception as e: errors.append("AKShare "+repr(e))
    try:
        j=requests.get("https://push2.eastmoney.com/api/qt/clist/get",params={"pn":1,"pz":300,"po":1,"np":1,"fltt":2,"invt":2,"fid":"f3","fs":"m:90+t:2+f:!50","fields":"f2,f3,f6,f8,f9,f12,f14,f23"},headers=UA,timeout=25).json()
        rows=[{"code":str(x.get("f12","")),"name":str(x.get("f14") or x.get("f12")),"close":num(x.get("f2")),"pct":num(x.get("f3")),"amount":num(x.get("f6")),"turnover":num(x.get("f8")),"pe":num(x.get("f9")),"pb":num(x.get("f23"))} for x in j.get("data",{}).get("diff",[])]
        if len(rows)>=20:return rows,"东方财富公开接口"
    except Exception as e: errors.append("Eastmoney "+repr(e))
    raise RuntimeError(" | ".join(errors))

def ranks(frame, col):
    s=pd.to_numeric(frame[col],errors="coerce")
    return (s.rank(pct=True)*100).fillna(50)

def calc_scores(frame):
    f=frame.copy()
    f["mom20"]=f.groupby("name")["close"].pct_change(20)
    f["flow5"]=f.groupby("name")["amount"].pct_change(5)
    # 每个交易日横截面排名；历史初始化时估值不可得，使用中性50分。
    f["price_score"]=f.groupby("date",group_keys=False).apply(lambda x:ranks(x,"mom20"))
    f["volume_score"]=f.groupby("date",group_keys=False).apply(lambda x:ranks(x,"amount"))
    f["turn_score"]=f.groupby("date",group_keys=False).apply(lambda x:ranks(x,"turnover"))
    f["flow_score"]=f.groupby("date",group_keys=False).apply(lambda x:ranks(x,"flow5"))
    f["valuation_score"]=50.0
    f["crowding"]=(.25*f["price_score"]+.20*f["volume_score"]+.20*f["turn_score"]+.20*f["flow_score"]+.15*f["valuation_score"]).clip(0,100)
    return f

def backfill_series(rows, days=180):
    end=datetime.date.today().strftime("%Y%m%d")
    all_rows=[]
    for idx,r in enumerate(rows):
        code=str(r.get("code",""))
        if not code.startswith("BK"): continue
        try:
            j=requests.get("https://push2his.eastmoney.com/api/qt/stock/kline/get",params={"secid":"90."+code,"fields1":"f1,f2,f3,f4,f5,f6","fields2":"f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61","klt":"101","fqt":"0","end":end,"lmt":days},headers=UA,timeout=15).json()
            for line in j.get("data",{}).get("klines",[]):
                p=line.split(",")
                if len(p)>=11:
                    all_rows.append({"date":p[0],"name":r["name"],"close":num(p[2]),"pct":num(p[8]),"amount":num(p[6]),"turnover":num(p[10])})
        except Exception:
            pass
        if idx%15==0: time.sleep(.15)
    if not all_rows:return []
    f=pd.DataFrame(all_rows).dropna(subset=["date","name","close"])
    f=calc_scores(f.sort_values(["name","date"]))
    return [{"date":str(x.date),"name":str(x.name),"crowding":round(finite(x.crowding),1),"close":round(finite(x.close,0),3),"pct":round(finite(x.pct,0),2)} for x in f.itertuples()]

def append_today_series(series, scored):
    today=str(scored["date"].iloc[-1])
    series=[x for x in series if x.get("date")!=today]
    for x in scored[scored["date"]==today].itertuples():
        series.append({"date":today,"name":str(x.name),"crowding":round(finite(x.crowding),1),"close":round(finite(x.close,0),3),"pct":round(finite(x.pct,0),2)})
    return series

today_date=datetime.date.today().isoformat()
try:
    rows,source=fetch_rows()
    today=pd.DataFrame(rows); today.insert(0,"date",today_date)
    hist=pd.read_csv(HIST) if HIST.exists() else pd.DataFrame(columns=today.columns)
    hist=hist[hist["date"].astype(str)!=today_date] if len(hist) else hist
    hist=pd.concat([hist,today],ignore_index=True,sort=False)
    hist.to_csv(HIST,index=False)

    # 完整拥挤度：当前日采用五因子；历史图表使用价格/成交/换手/资金四因子+估值中性重建。
    c=today.set_index("name")
    out=[]
    for name,g in hist.groupby("name"):
        g=g.sort_values("date").tail(750); z=g.iloc[-1]
        def hr(s,min_n=20):
            s=pd.to_numeric(pd.Series(s),errors="coerce").dropna()
            return s.rank(pct=True).iloc[-1]*100 if len(s)>=min_n else np.nan
        def nowrank(col):
            s=pd.to_numeric(c[col],errors="coerce")
            return finite(s.rank(pct=True).get(name,50)*100,50)
        price=hr(g["close"].pct_change(20)); price=nowrank("pct") if not math.isfinite(finite(price,np.nan)) else price
        volume=hr(g["amount"]); volume=nowrank("amount") if not math.isfinite(finite(volume,np.nan)) else volume
        turnover=hr(g["turnover"]); turnover=nowrank("turnover") if not math.isfinite(finite(turnover,np.nan)) else turnover
        flow=hr(g["amount"].pct_change(5)); flow=nowrank("pct") if not math.isfinite(finite(flow,np.nan)) else flow
        valuation=np.nanmean([hr(g["pe"]),hr(g["pb"])])
        if not math.isfinite(finite(valuation,np.nan)):
            valuation=np.nanmean([nowrank("pe"),nowrank("pb")])
        price,volume,turnover,flow,valuation=[finite(v,50) for v in (price,volume,turnover,flow,valuation)]
        score=round(float(np.clip(.25*price+.20*volume+.20*turnover+.20*flow+.15*valuation,0,100)),1)
        opp=round(float(np.clip(.55*(100-score)+.25*price+.20*(100-valuation),0,100)),1)
        risk="极度拥挤" if score>=85 else "高拥挤" if score>=70 else "中度拥挤" if score>=55 else "中性" if score>=40 else "低拥挤"
        change20=0.0
        if len(g)>=21 and num(g.iloc[-21]["close"]) not in (0,np.nan):
            base=num(g.iloc[-21]["close"]); cur=num(z["close"])
            if pd.notna(base) and pd.notna(cur) and base!=0: change20=round((cur/base-1)*100,1)
        out.append({"name":name,"crowding":score,"change20":change20,"risk":risk,"opportunity":opp})
    out.sort(key=lambda x:x["crowding"],reverse=True)

    # 首次自动初始化180个交易日前后的历史图表数据；以后每天只追加当天快照。
    series=json.loads(SERIES.read_text(encoding="utf-8")) if SERIES.exists() else []
    if len(series)<500:
        series=backfill_series(rows,180)
    # 用当天真实五因子分数覆盖历史重建的最后一天
    for x in out:
        series=[v for v in series if not (v.get("date")==today_date and v.get("name")==x["name"])]
        raw=c.loc[x["name"]]
        series.append({"date":today_date,"name":x["name"],"crowding":x["crowding"],"close":round(finite(raw["close"],0),3),"pct":round(finite(raw["pct"],0),2)})
    series.sort(key=lambda x:(x["date"],x["name"]))
    SERIES.write_text(json.dumps(series,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")

    days=hist["date"].nunique()
    top=[x["name"] for x in out[:3]]
    opps=sorted(out,key=lambda x:x["opportunity"],reverse=True)[:3]
    quality=("五因子实时评分｜图表历史初始化完成" if len(series)>=500 else "初始化阶段｜历史数据继续积累")
    result={"date":today_date,"status":"云端自动更新成功","data_quality":quality+"｜数据源："+source,"summary":"当前拥挤度较高："+ "、".join(top)+"。机会雷达关注："+ "、".join(x["name"] for x in opps)+"。","industries":out,"history_points":len(series)}
except Exception as e:
    result={"date":today_date,"status":"数据更新失败："+repr(e)[:180],"data_quality":"--","summary":"本次数据更新失败。","industries":[],"history_points":0}

OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
print(result["status"],"industries:",len(result["industries"]), "history:",result["history_points"])
