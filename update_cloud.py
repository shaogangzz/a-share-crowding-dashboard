import json,datetime,os
from pathlib import Path
import numpy as np,pandas as pd
DATA=Path("data"); DATA.mkdir(exist_ok=True)
HIST=DATA/"history.csv"; OUT=DATA/"latest.json"
def pct_rank(s):
    s=pd.Series(s).dropna()
    return float(s.rank(pct=True).iloc[-1]*100) if len(s)>=20 else np.nan
try:
 import akshare as ak
 spot=ak.stock_board_industry_spot_em()
 name_col="板块名称" if "板块名称" in spot.columns else "名称"
 def n(v):
  try:return float(str(v).replace(",","").replace("%",""))
  except:return np.nan
 rows=[]
 for _,r in spot.iterrows():
  name=str(r[name_col])
  rows.append({"date":datetime.date.today().isoformat(),"name":name,"close":n(r.get("最新价")),"pct":n(r.get("涨跌幅")),"turnover":n(r.get("换手率")),"amount":n(r.get("成交额")),"pe":n(r.get("市盈率-动态")),"pb":n(r.get("市净率"))})
 today=pd.DataFrame(rows)
 hist=pd.read_csv(HIST) if HIST.exists() else pd.DataFrame(columns=today.columns)
 hist=pd.concat([hist[hist.date!=today.date] if len(hist) else hist,today],ignore_index=True)
 hist.to_csv(HIST,index=False)
 out=[]
 for name,g in hist.groupby("name"):
  g=g.sort_values("date").tail(750)
  z=g.iloc[-1]
  ret20=(z.close/g.close.iloc[-21]-1)*100 if len(g)>=21 and pd.notna(z.close) and pd.notna(g.close.iloc[-21]) else np.nan
  ret60=(z.close/g.close.iloc[-61]-1)*100 if len(g)>=61 and pd.notna(z.close) and pd.notna(g.close.iloc[-61]) else np.nan
  price=np.nanmean([pct_rank(g.assign(v=g.close.pct_change(20)).v),pct_rank(g.assign(v=g.close.pct_change(60)).v)])
  volume=pct_rank(g.amount)
  turnover=pct_rank(g.turnover)
  valuation=np.nanmean([pct_rank(g.pe),pct_rank(g.pb)])
  # 免费公开源资金流覆盖不稳定：暂以成交额趋势代理，并保留因子扩展位
  flow=pct_rank(g.amount.pct_change(5))
  vals=[(price,.25),(volume,.20),(turnover,.20),(flow,.20),(valuation,.15)]
  valid=[(v,w) for v,w in vals if pd.notna(v)]
  score=sum(v*w for v,w in valid)/sum(w for v,w in valid) if valid else np.nan
  opp=np.nanmean([100-score if pd.notna(score) else np.nan,100-(valuation if pd.notna(valuation) else 50),price if pd.notna(price) else 50])
  risk="极度拥挤" if score>=85 else "高拥挤" if score>=70 else "中度拥挤" if score>=55 else "中性" if score>=40 else "低拥挤"
  out.append({"name":name,"crowding":round(float(score),1) if pd.notna(score) else 50.0,"change20":round(float(ret20 or 0),1),"risk":risk,"opportunity":round(float(opp),1) if pd.notna(opp) else 50.0})
 out=sorted(out,key=lambda x:x["crowding"],reverse=True)
 high=[x["name"] for x in out[:3]]; low=sorted(out,key=lambda x:x["opportunity"],reverse=True)[:3]
 summary=f"当前拥挤度较高的方向包括：{'、'.join(high)}。机会雷达优先关注：{'、'.join(x['name'] for x in low)}。高拥挤并不等于立即下跌，但应警惕资金撤离和波动放大。"
 result={"date":datetime.date.today().isoformat(),"status":"云端自动更新成功","data_quality":"公开数据源自动计算","summary":summary,"industries":out}
except Exception as e:
 old=json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"industries":[]}
 result=old; result["date"]=datetime.date.today().isoformat(); result["status"]="数据源更新失败，保留上一份有效数据"
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
