import pandas as pd, numpy as np, json, joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv('/mnt/user-data/uploads/sales_forecasting_dataset_1500__3___1_.csv',parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
for c in ['Discount_Percent','Marketing_Spend','Competitor_Price','Inventory_Level']:
    df[c]=df.groupby('Product_Category')[c].transform(lambda x:x.fillna(x.median()))
cats=sorted(df.Product_Category.unique()); regs=sorted(df.Region.unique())
days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['Cat']=df.Product_Category.map({c:i for i,c in enumerate(cats)}); df['Reg']=df.Region.map({c:i for i,c in enumerate(regs)})
df['Day']=df.Day_of_Week.map({d:i for i,d in enumerate(days)})
df['Promo']=(df.Promotion=='Yes')*1; df['Hol']=(df.Holiday=='Yes')*1
df['Price_Gap']=df.Unit_Price-df.Competitor_Price
F=['Unit_Price','Discount_Percent','Promo','Hol','Marketing_Spend','Competitor_Price','Price_Gap','Inventory_Level','Month','Quarter','Day','Cat','Reg']
print(df[['Sales','Units_Sold','Unit_Price','Customer_Count']].corr().round(2).iloc[0])
n=int(len(df)*.8); tr,te=df.iloc[:n],df.iloc[n:]
grid={'n_estimators':[100,200],'learning_rate':[0.05,0.1],'max_depth':[2,3,4],'subsample':[0.8,1.0]}
gs=GridSearchCV(GradientBoostingRegressor(random_state=42),grid,cv=TimeSeriesSplit(5),scoring='neg_root_mean_squared_error',n_jobs=-1).fit(tr[F],tr.Sales)
m=gs.best_estimator_; p=m.predict(te[F])
base=GradientBoostingRegressor(random_state=42).fit(tr[F],tr.Sales).predict(te[F])
met=lambda y,p:dict(MAE=mean_absolute_error(y,p),RMSE=mean_squared_error(y,p)**.5,R2=r2_score(y,p))
res=pd.DataFrame(gs.cv_results_)
top=[dict(params=r.params,rmse=-r.mean_test_score,rank=int(r.rank_test_score)) for r in res.sort_values('rank_test_score').head(8).itertuples()]
def tree(t):
    return dict(l=t.children_left.tolist(),r=t.children_right.tolist(),f=t.feature.tolist(),t=[round(float(x),5) for x in t.threshold],v=[round(float(x),4) for x in t.value[:,0,0]])
model=dict(init=float(m.init_.constant_[0][0]),lr=m.learning_rate,trees=[tree(e[0].tree_) for e in m.estimators_],features=F)
meta=dict(features=F,cats=cats,regs=regs,days=days,best=gs.best_params_,cv_rmse=-gs.best_score_,
 tuned=met(te.Sales,p),baseline=met(te.Sales,base),n_train=len(tr),n_test=len(te),grid_size=len(res),
 importance=sorted(zip(F,m.feature_importances_.round(4).tolist()),key=lambda x:-x[1]),top=top,
 ranges={c:[float(df[c].min()),float(df[c].max()),float(df[c].median())] for c in ['Unit_Price','Discount_Percent','Marketing_Spend','Competitor_Price','Inventory_Level']},
 test=[dict(d=str(d.date()),a=round(a,1),p=round(q,1)) for d,a,q in zip(te.Date,te.Sales,p)][::3],
 monthly=df.groupby(df.Date.dt.to_period('M')).Sales.sum().round(0).reset_index().assign(Date=lambda x:x.Date.astype(str)).values.tolist())
json.dump(dict(model=model,meta=meta),open('bundle.json','w'),default=float,separators=(',',':'))
joblib.dump(m,'model.joblib'); json.dump(meta,open('meta.json','w'),default=float)
print(meta['best'],meta['tuned'],meta['baseline'],meta['importance'][:5],len(open('bundle.json').read())//1024,'KB')
