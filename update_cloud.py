import json, datetime, math, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA=Path("data"); DATA.mkdir(exist_ok=True)
HIST=DATA/"history.csv"; OUT=DATA/"latest.json"; SERIES=DATA/"timeseries.json"
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36","Referer":"https://quote.eastmoney.com/","Accept":"application/json,text/plain,*/*","Accept-Language":"zh-CN,zh;q=0.9"}

def session():
    s=requests.Session()
    s.headers.update(UA)
    s.mount("https://",HTTPAdapter(max_retries=Retry(total=4,connect=4,read=4,backoff_factor=1,status_forcelist=[429,500,502,503,504],allowed_methods=False)))
    return s

HTTP=session()

def num(v):
    try:
        if v is None:return np.nan
        x=float(str(v).replace(",","").replace("%","").strip())
        return x if math.isfinite(x) else np.nan
    except:return np.nan

def finite(v,default=50.0):
    try:
        x=float(v);return x if math.isfinite(x) else default
    except:return default

def east_json(url,params):
    last=None
    for n in range(5):
        try:
            r=HTTP.get(url,params=params,timeout=(10,35))
            if r.status_code!=200: raise RuntimeError("HTTP "+str(r.status_code))
            if not r.text or r.text.lstrip()[:1] not in "{[": raise RuntimeError("non-json response: "+r.text[:80])
            return r.json()
        except Exception as e:
            last=e; time.sleep(1.5+n*1.5+random.random())
    raise last

def fetch_rows():
    errors=[]
    # 先直连东财，避免 AKShare 自身版本/连接问题
    params={"pn":1,"pz":500,"po":1,"np":1,"fltt":2,"invt":2,"fid":"f3","fs":"m:90+t:2+f:!50","fields":"f2,f3,f6,f8,f9,f12,f14,f23","_":int(time.time()*1000)}
    for host in ["https://push2.eastmoney.com/api/qt/clist/get","https://push2delay.eastmoney.com/api/qt/clist/get"]:
        try:
            j=east_json(host,params)
            diff=j.get("data",{}).get("diff",[])
            rows=[{"code":str(x.get("f12","")),"name":str(x.get("f14") or x.get("f12")),"close":num(x.get("f2")),"pct":num(x.get("f3")),"amount":num(x.get("f6")),"turnover":num(x.get("f8")),"pe":num(x.get("f9")),"pb":num(x.get("f23"))} for x in diff if x.get("f14")]
            if len(rows)>=20:return rows,"东方财富公开行情"
            errors.append(host+" rows="+str(len(rows)))
        except Exception as e:errors.append(host+" "+repr(e)[:120])
    # 再尝试 AKShare，多次重试
    try:
        import akshare as ak
        for n in range(3):
            try:
                df=ak.stock_board_industry_spot_em()
                nc="板块名称" if "板块名称" in df.columns else "名称"
                rows=[{"code":str(r.get("板块代码","")),"name":str(r[nc]),"close":num(r.get("最新价")),"pct":num(r.get("涨跌幅")),"amount":num(r.get("成交额")),"turnover":num(r.get("换手率")),"pe":num(r.get("市盈率-动态")),"pb":num(r.get("市净率"))} for _,r in df.iterrows()]
                if len(rows)>=20:return rows,"AKShare/东方财富"
            except Exception as e:
                errors.append("AKShare#"+str(n+1)+" "+repr(e)[:120]);time.sleep(3+n*2)
    except Exception as e:errors.append("AKShare import "+repr(e))
    raise RuntimeError(" | ".join(errors))

def ranks(s):
    return pd.to_numeric(s,errors="coerce").rank(pct=True).fillna(.5)*100

def historical_scores(df):
    f=df.copy().sort_values(["name","date"])
    f["mom20"]=f.groupby("name")["close"].pct_change(20)
    f["flow5"]=f.groupby("name")["amount"].pct_change(5)
    for col,new in [("mom20","price_score"),("amount","volume_score"),("turnover","turn_score"),("flow5","flow_score")]:
        f[new]=f.groupby("date",group_keys=False)[col].apply(ranks)
    f["crowding"]=(.25*f["price_score"]+.20*f["volume_score"]+.20*f["turn_score"]+.20*f["flow_score"]+7.5).clip(0,100)
    return f

def backfill(rows,days=180):
    end=datetime.date.today().strftime("%Y%m%d");allrows=[]
    for i,r in enumerate(rows):
        code=str(r.get("code",""))
        if not code.startswith("BK"):continue
        try:
            j=east_json("https://push2his.eastmoney.com/api/qt/stock/kline/get",{"secid":"90."+code,"fields1":"f1,f2,f3,f4,f5,f6","fields2":"f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61","klt":"101","fqt":"0","end":end,"lmt":days,"_":int(time.time()*1000)})
            for line in j.get("data",{}).get("klines",[]):
                p=line.split(",")
                if len(p)>=11:allrows.append({"date":p[0],"name":r["name"],"close":num(p[2]),"pct":num(p[8]),"amount":num(p[6]),"turnover":num(p[10])})
        except Exception:pass
        if i%10==0:time.sleep(.3)
    if not allrows:return []
    f=historical_scores(pd.DataFrame(allrows).dropna(subset=["close"]).sort_values(["name","date"]))
    return [{"date":str(x.date),"name":str(x.name),"crowding":round(finite(x.crowding),1),"close":round(finite(x.close,0),3),"pct":round(finite(x.pct,0),2)} for x in f.itertuples()]

def save_result(result):
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

today_date=datetime.date.today().isoformat()
previous=None
try:
    if OUT.exists():previous=json.loads(OUT.read_text(encoding="utf-8"))
except:previous=None

try:
    rows,source=fetch_rows()
    today=pd.DataFrame(rows);today.insert(0,"date",today_date)
    hist=pd.read_csv(HIST) if HIST.exists() else pd.DataFrame(columns=today.columns)
    hist=hist[hist["date"].astype(str)!=today_date] if len(hist) else hist
    hist=pd.concat([hist,today],ignore_index=True,sort=False);hist.to_csv(HIST,index=False)
    c=today.set_index("name");out=[]
    for name,g in hist.groupby("name"):
        g=g.sort_values("date").tail(750)
        def hr(s,minn=20):
            s=pd.to_numeric(pd.Series(s),errors="coerce").dropna()
            return s.rank(pct=True).iloc[-1]*100 if len(s)>=minn else np.nan
        def nr(col):
            return finite(pd.to_numeric(c[col],errors="coerce").rank(pct=True).get(name,50)*100,50)
        vals=[hr(g["close"].pct_change(20)),hr(g["amount"]),hr(g["turnover"]),hr(g["amount"].pct_change(5))]
        vals=[nr("pct") if not math.isfinite(finite(vals[0],np.nan)) else vals[0],nr("amount") if not math.isfinite(finite(vals[1],np.nan)) else vals[1],nr("turnover") if not math.isfinite(finite(vals[2],np.nan)) else vals[2],nr("pct") if not math.isfinite(finite(vals[3],np.nan)) else vals[3]]
        val=np.nanmean([hr(g["pe"]),hr(g["pb"])])
        if not math.isfinite(finite(val,np.nan)):val=np.nanmean([nr("pe"),nr("pb")])
        p,v,t,fl,va=[finite(x,50) for x in vals+[val]]
        score=round(float(np.clip(.25*p+.20*v+.20*t+.20*fl+.15*va,0,100)),1)
        opp=round(float(np.clip(.55*(100-score)+.25*p+.20*(100-va),0,100)),1)
        risk="极度拥挤" if score>=85 else "高拥挤" if score>=70 else "中度拥挤" if score>=55 else "中性" if score>=40 else "低拥挤"
        ch=0.0
        if len(g)>=21:
            base,cur=num(g.iloc[-21]["close"]),num(g.iloc[-1]["close"])
            if pd.notna(base) and pd.notna(cur) and base!=0:ch=round((cur/base-1)*100,1)
        out.append({"name":name,"crowding":score,"change20":ch,"risk":risk,"opportunity":opp})
    out.sort(key=lambda x:x["crowding"],reverse=True)
    series=json.loads(SERIES.read_text(encoding="utf-8")) if SERIES.exists() else []
    if len(series)<500:
        seeded=backfill(rows,180)
        if seeded:series=seeded
    series=[x for x in series if x.get("date")!=today_date]
    raw={str(r["name"]):r for _,r in today.iterrows()}
    series += [{"date":today_date,"name":x["name"],"crowding":x["crowding"],"close":round(finite(raw[x["name"]]["close"],0),3),"pct":round(finite(raw[x["name"]]["pct"],0),2)} for x in out if x["name"] in raw]
    series.sort(key=lambda x:(x["date"],x["name"]))
    SERIES.write_text(json.dumps(series,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    top=[x["name"] for x in out[:3]];opps=sorted(out,key=lambda x:x["opportunity"],reverse=True)[:3]
    result={"date":today_date,"status":"云端自动更新成功","data_quality":("五因子实时评分｜历史数据已初始化" if len(series)>=500 else "五因子实时评分｜历史数据持续积累")+"｜数据源："+source,"summary":"当前拥挤度较高："+ "、".join(top)+"。机会雷达关注："+ "、".join(x["name"] for x in opps)+"。","industries":out,"history_points":len(series)}
    save_result(result);print(result["status"],len(out),len(series))
except Exception as e:
    # 核心原则：失败绝不清空上一份有效数据
    if previous and previous.get("industries"):
        previous["status"]="本次更新失败，展示上一份有效数据｜"+repr(e)[:100]
        previous["data_quality"]=(previous.get("data_quality") or "")+"｜已启用数据保护"
        save_result(previous);print(previous["status"])
    else:
        result={"date":today_date,"status":"数据源暂时不可用，系统将在下次自动任务重试："+repr(e)[:160],"data_quality":"等待首次成功抓取","summary":"本次未获得有效行情数据，未生成任何虚假数据。","industries":[],"history_points":0}
        save_result(result);print(result["status"])
