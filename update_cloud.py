import json, datetime, math, time, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA=Path("data"); DATA.mkdir(exist_ok=True)
HIST=DATA/"history.csv"; OUT=DATA/"latest.json"; SERIES=DATA/"timeseries.json"
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36","Referer":"https://quote.eastmoney.com/","Accept":"application/json,text/plain,*/*","Accept-Language":"zh-CN,zh;q=0.9"}

def make_session():
    s=requests.Session(); s.headers.update(UA)
    s.mount("https://",HTTPAdapter(max_retries=Retry(total=2,connect=2,read=2,backoff_factor=.5,status_forcelist=[429,500,502,503,504],allowed_methods=False)))
    return s
HTTP=make_session()

def num(v):
    try:
        if v is None:return np.nan
        x=float(str(v).replace(",","").replace("%","").strip()); return x if math.isfinite(x) else np.nan
    except:return np.nan

def finite(v,default=50.0):
    try:
        x=float(v); return x if math.isfinite(x) else default
    except:return default

def east_json(url,params,attempts=3):
    last=None
    for n in range(attempts):
        try:
            r=HTTP.get(url,params=params,timeout=(6,15)); r.raise_for_status()
            if not r.text or r.text.lstrip()[:1] not in "{[": raise RuntimeError("non-json")
            return r.json()
        except Exception as e:
            last=e
            if n+1<attempts: time.sleep(.6*(n+1)+random.random()*.4)
    raise last

def fetch_rows():
    params={"pn":1,"pz":500,"po":1,"np":1,"fltt":2,"invt":2,"fid":"f3","fs":"m:90+t:2+f:!50","fields":"f2,f3,f6,f8,f9,f12,f14,f23","_":int(time.time()*1000)}
    errors=[]
    for host in ["https://push2.eastmoney.com/api/qt/clist/get","https://push2delay.eastmoney.com/api/qt/clist/get"]:
        try:
            j=east_json(host,params); diff=j.get("data",{}).get("diff",[])
            rows=[{"code":str(x.get("f12","")),"name":str(x.get("f14") or x.get("f12")),"close":num(x.get("f2")),"pct":num(x.get("f3")),"amount":num(x.get("f6")),"turnover":num(x.get("f8")),"pe":num(x.get("f9")),"pb":num(x.get("f23"))} for x in diff if x.get("f14")]
            if len(rows)>=20:return rows,"东方财富公开行情"
        except Exception as e: errors.append(repr(e)[:100])
    try:
        import akshare as ak
        df=ak.stock_board_industry_spot_em(); nc="板块名称" if "板块名称" in df.columns else "名称"
        rows=[{"code":str(r.get("板块代码","")),"name":str(r[nc]),"close":num(r.get("最新价")),"pct":num(r.get("涨跌幅")),"amount":num(r.get("成交额")),"turnover":num(r.get("换手率")),"pe":num(r.get("市盈率-动态")),"pb":num(r.get("市净率"))} for _,r in df.iterrows()]
        if len(rows)>=20:return rows,"AKShare/东方财富"
    except Exception as e: errors.append(repr(e)[:100])
    raise RuntimeError("实时行情获取失败: "+" | ".join(errors))

def ranks(s): return pd.to_numeric(s,errors="coerce").rank(pct=True).fillna(.5)*100

def historical_scores(df):
    f=df.copy().sort_values(["name","date"])
    f["mom20"]=f.groupby("name")["close"].pct_change(20); f["flow5"]=f.groupby("name")["amount"].pct_change(5)
    for col,new in [("mom20","price_score"),("amount","volume_score"),("turnover","turn_score"),("flow5","flow_score")]: f[new]=f.groupby("date",group_keys=False)[col].apply(ranks)
    f["crowding"]=(.25*f["price_score"]+.20*f["volume_score"]+.20*f["turn_score"]+.20*f["flow_score"]+7.5).clip(0,100)
    return f

def fetch_one(r,days=260):
    name=str(r.get("name","")); code=str(r.get("code",""))
    if not code.startswith("BK"): return name,None
    try:
        j=east_json("https://push2his.eastmoney.com/api/qt/stock/kline/get",{"secid":"90."+code,"fields1":"f1,f2,f3,f4,f5,f6","fields2":"f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61","klt":"101","fqt":"0","end":datetime.date.today().strftime("%Y%m%d"),"lmt":days,"_":int(time.time()*1000)},attempts=2)
        ks=j.get("data",{}).get("klines",[])
        if len(ks)<200:return name,None
        rows=[]
        for line in ks:
            p=line.split(",")
            if len(p)>=11: rows.append({"date":p[0],"name":name,"close":num(p[2]),"pct":num(p[8]),"amount":num(p[6]),"turnover":num(p[10]),"pe":np.nan,"pb":np.nan})
        return name,pd.DataFrame(rows) if len(rows)>=200 else None
    except:return name,None

def backfill(rows):
    """并发抓取：8线程、每行业最多2次尝试；目标是秒级到几分钟，而非逐个串行十几分钟。"""
    parts=[]; ok=[]; failed=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs=[ex.submit(fetch_one,r,260) for r in rows]
        for fut in as_completed(futs):
            name,df=fut.result()
            if df is not None: parts.append(df); ok.append(name)
            else: failed.append(name)
    return (pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()),sorted(ok),sorted(failed)

def coverage_map(series):
    out={}
    for x in series:
        if x.get("name") and x.get("date"): out.setdefault(x["name"],set()).add(str(x["date"]))
    return {k:len(v) for k,v in out.items()}

def save_result(x): OUT.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

today_date=datetime.date.today().isoformat(); previous=None
try:
    if OUT.exists(): previous=json.loads(OUT.read_text(encoding="utf-8"))
except:pass
try:
    rows,source=fetch_rows(); today=pd.DataFrame(rows); today.insert(0,"date",today_date)
    hist=pd.read_csv(HIST) if HIST.exists() else pd.DataFrame(columns=today.columns)
    if len(hist): hist=hist[hist["date"].astype(str)!=today_date]
    pd.concat([hist,today],ignore_index=True,sort=False).to_csv(HIST,index=False)
    hist=pd.read_csv(HIST); c=today.set_index("name"); out=[]
    for name,g in hist.groupby("name"):
        g=g.sort_values("date").tail(750)
        def hr(s):
            s=pd.to_numeric(pd.Series(s),errors="coerce").dropna(); return s.rank(pct=True).iloc[-1]*100 if len(s)>=20 else np.nan
        def nr(col):
            s=pd.to_numeric(c[col],errors="coerce")
            ranked=s.rank(pct=True)
            x=ranked.get(name,np.nan)
            return float(x*100) if pd.notna(x) and math.isfinite(float(x)) else 50.0
        vals=[hr(g["close"].pct_change(20)),hr(g["amount"]),hr(g["turnover"]),hr(g["amount"].pct_change(5))]
        vals=[nr("pct") if not math.isfinite(finite(v,np.nan)) else v for v in vals]
        val=np.nanmean([hr(g["pe"]),hr(g["pb"])]); val=np.nanmean([nr("pe"),nr("pb")]) if not math.isfinite(finite(val,np.nan)) else val
        p,v,t,fl,va=[finite(x,50) for x in vals+[val]]
        score=round(float(np.clip(.25*p+.20*v+.20*t+.20*fl+.15*va,0,100)),1); opp=round(float(np.clip(.55*(100-score)+.25*p+.20*(100-va),0,100)),1)
        risk="极度拥挤" if score>=85 else "高拥挤" if score>=70 else "中度拥挤" if score>=55 else "中性" if score>=40 else "低拥挤"
        ch=0
        if len(g)>=21:
            a,b=num(g.iloc[-21]["close"]),num(g.iloc[-1]["close"]); ch=round((b/a-1)*100,1) if pd.notna(a) and pd.notna(b) and a else 0
        out.append({"name":name,"crowding":score,"change20":ch,"risk":risk,"opportunity":opp})
    out.sort(key=lambda x:x["crowding"],reverse=True)

    # 一次并发完成全部行业历史初始化；不再分批等待多天
    rawdf,ok,failed=backfill(rows)
    if rawdf.empty or len(ok)<max(20,int(len(rows)*.7)): raise RuntimeError(f"历史数据获取不足：{len(ok)}/{len(rows)}")
    scored=historical_scores(rawdf.drop_duplicates(["name","date"],keep="last"))
    series=[{"date":str(x.date),"name":str(x.name),"crowding":round(finite(x.crowding),1),"close":round(finite(x.close,0),3),"pct":round(finite(x.pct,0),2)} for x in scored.itertuples()]
    series=[x for x in series if x["date"]!=today_date]
    raw={str(r["name"]):r for _,r in today.iterrows()}
    series += [{"date":today_date,"name":x["name"],"crowding":x["crowding"],"close":round(finite(raw[x["name"]]["close"],0),3),"pct":round(finite(raw[x["name"]]["pct"],0),2)} for x in out if x["name"] in raw]
    series.sort(key=lambda x:(x["date"],x["name"])); SERIES.write_text(json.dumps(series,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    cov=coverage_map(series); complete=sum(1 for r in rows if cov.get(str(r["name"]),0)>=200)
    top=[x["name"] for x in out[:3]]; opps=sorted(out,key=lambda x:x["opportunity"],reverse=True)[:3]
    result={"date":today_date,"status":"云端自动更新成功" if complete==len(rows) else "云端更新成功，少量行业待重试","data_quality":f"五因子实时评分｜历史完整 {complete}/{len(rows)} 个行业≥200交易日｜本次成功 {len(ok)}/{len(rows)}｜并发8线程｜数据源：{source}","summary":"当前拥挤度较高："+"、".join(top)+"。机会雷达关注："+"、".join(x["name"] for x in opps)+"。"+(f"仍有{len(rows)-complete}个行业待重试。" if complete<len(rows) else "全部行业历史数据已完成初始化。"),"industries":out,"history_points":len(series),"history_progress":{"complete":complete,"total":len(rows),"missing":len(rows)-complete,"failed":failed}}
    save_result(result); print(result["status"],len(out),len(series),"complete",complete)
except Exception as e:
    if previous and previous.get("industries"):
        previous["status"]="本次更新失败，展示上一份有效数据｜"+repr(e)[:120]; save_result(previous); print(previous["status"])
    else:
        save_result({"date":today_date,"status":"数据源暂时不可用："+repr(e)[:160],"data_quality":"等待首次成功抓取","summary":"本次未获得有效行情数据，未生成虚假数据。","industries":[],"history_points":0}); print("failed",e)
