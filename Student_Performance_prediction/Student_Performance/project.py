"""CA1: Student Performance Prediction — Units I & II.
Run: python project.py   (Python 3.10+, pip install -r requirements.txt)
Dataset: Student_Performance.csv supplied with the uploaded GitHub project.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib, json
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
BASE=Path(__file__).resolve().parent
OUT=BASE/'results'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(BASE/'Student_Performance.csv')
print('Shape:',df.shape,'\nMissing values:\n',df.isna().sum(),'\nSummary:\n',df.describe().round(2))
# Encoding categorical input; never scale the target.
df['Extracurricular Activities']=df['Extracurricular Activities'].map({'No':0,'Yes':1})
assert not df.isna().any().any(), 'Missing or unexpected values detected'
X=df.drop(columns='Performance Index'); y=df['Performance Index']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42)
cv=KFold(n_splits=5,shuffle=True,random_state=42)
models={
 'Simple Linear Regression':Pipeline([('scale',StandardScaler()),('model',LinearRegression())]),
 'Multiple Linear Regression':Pipeline([('scale',StandardScaler()),('model',LinearRegression())]),
 'Polynomial Regression':Pipeline([('scale',StandardScaler()),('poly',PolynomialFeatures(degree=2,include_bias=False)),('model',Ridge(alpha=1.0))]),
 'Ridge Regression':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=1.0))]),
 'Lasso Regression':Pipeline([('scale',StandardScaler()),('model',Lasso(alpha=1.0,max_iter=30000))]),
}
# Simple regression uses only one predictor; all other models use all five.
searches={
 'Polynomial Regression':{'poly__degree':[2,3],'model__alpha':[0.1,1,10,100]},
 'Ridge Regression':{'model__alpha':[0.01,0.1,1,10,100,1000]},
 'Lasso Regression':{'model__alpha':[0.001,0.01,0.1,1,10]},
}
rows=[]; trained={}; predictions={}; tuning={}
for name,pipe in models.items():
    Xi=X_train[['Hours Studied']] if name.startswith('Simple') else X_train
    Xt=X_test[['Hours Studied']] if name.startswith('Simple') else X_test
    if name in searches:
        gs=GridSearchCV(pipe,searches[name],cv=cv,scoring='neg_root_mean_squared_error',n_jobs=-1)
        gs.fit(Xi,y_train); estimator=gs.best_estimator_; tuning[name]=gs.best_params_
    else: estimator=pipe
    scores=cross_validate(estimator,Xi,y_train,cv=cv,scoring={'rmse':'neg_root_mean_squared_error','r2':'r2'},n_jobs=-1)
    estimator.fit(Xi,y_train)
    pred=estimator.predict(Xt); train_pred=estimator.predict(Xi)
    rows.append({'Model':name,'CV RMSE':-scores['test_rmse'].mean(),'CV R2':scores['test_r2'].mean(),
       'Train RMSE':np.sqrt(mean_squared_error(y_train,train_pred)),
       'Test MAE':mean_absolute_error(y_test,pred),'Test MSE':mean_squared_error(y_test,pred),
       'Test RMSE':np.sqrt(mean_squared_error(y_test,pred)),'Test R2':r2_score(y_test,pred)})
    trained[name]=estimator; predictions[name]=pred
results=pd.DataFrame(rows).sort_values('CV RMSE')
results.to_csv(OUT/'model_comparison.csv',index=False)
selected=results.iloc[0]['Model']; best=trained[selected]
joblib.dump({'model':best,'features':(['Hours Studied'] if selected.startswith('Simple') else list(X.columns)),'selected_name':selected},OUT/'best_model.joblib')
(OUT/'best_hyperparameters.json').write_text(json.dumps(tuning,indent=2))
print('\nMODEL RESULTS\n',results.round(4).to_string(index=False));print('\nBest by CV RMSE:',selected,'\nTuning:',tuning)
# Plots
plt.figure(figsize=(9,5)); plt.barh(results['Model'],results['CV RMSE']);plt.xlabel('5-fold CV RMSE (lower is better)');plt.title('Model comparison (training CV only)');plt.tight_layout();plt.savefig(OUT/'cv_model_comparison.png',dpi=150);plt.close()
plt.figure(figsize=(6,6));plt.scatter(y_test,predictions[selected],s=12,alpha=.5);lo=min(y_test.min(),predictions[selected].min());hi=max(y_test.max(),predictions[selected].max());plt.plot([lo,hi],[lo,hi],'--');plt.xlabel('Actual Performance Index');plt.ylabel('Predicted Performance Index');plt.title('Actual vs predicted: '+selected);plt.tight_layout();plt.savefig(OUT/'actual_vs_predicted.png',dpi=150);plt.close()
plt.figure(figsize=(7,4));plt.scatter(predictions[selected],y_test-predictions[selected],s=12,alpha=.5);plt.axhline(0,linestyle='--');plt.xlabel('Predictions');plt.ylabel('Residuals');plt.title('Residual plot');plt.tight_layout();plt.savefig(OUT/'residual_plot.png',dpi=150);plt.close()
# Complexity illustration: polynomial degree 1 through 4; separate from model selection.
complexity=[]
for degree in [1,2,3,4]:
    p=Pipeline([('scale',StandardScaler()),('poly',PolynomialFeatures(degree=degree,include_bias=False)),('model',Ridge(alpha=10))])
    p.fit(X_train,y_train)
    c=cross_validate(p,X_train,y_train,cv=cv,scoring='neg_root_mean_squared_error',n_jobs=-1)
    complexity.append({'Degree':degree,'Train RMSE':np.sqrt(mean_squared_error(y_train,p.predict(X_train))),'CV RMSE':-c['test_score'].mean()})
pd.DataFrame(complexity).to_csv(OUT/'complexity_analysis.csv',index=False)
plt.figure(figsize=(7,4));plt.plot([d['Degree'] for d in complexity],[d['Train RMSE'] for d in complexity],marker='o',label='Train');plt.plot([d['Degree'] for d in complexity],[d['CV RMSE'] for d in complexity],marker='o',label='CV');plt.xlabel('Polynomial degree');plt.ylabel('RMSE');plt.legend();plt.title('Model complexity and generalization');plt.tight_layout();plt.savefig(OUT/'bias_variance_demo.png',dpi=150);plt.close()
# Coefficients of the standardized linear model
lin=trained['Multiple Linear Regression'].named_steps['model']
pd.DataFrame({'Feature':X.columns,'Standardized coefficient':lin.coef_}).to_csv(OUT/'linear_coefficients.csv',index=False)
print('\nFiles saved to:',OUT)
