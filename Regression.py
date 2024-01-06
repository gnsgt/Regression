    #Basit Dogrusal Regresyon
import pandas as pd
ad = pd.read_csv("Advertising.csv",usecols=(1,2,3,4)) #0.  Index " id'ler oldugu icin 0'i kullanmiyoruz
#df = df.iloc[:,1:len(df)] #Boyle de yapabilirdik
df = ad.copy()
df.isnull().values.any() #Eksik deger var mi
df.corr() #Korelasyon, Ornek: TV arttikca satis da artar
import seaborn as sns
sns.pairplot(df,kind="reg")
sns.jointplot(x = "TV", y = "sales",data = df,kind ="reg") #Tv ve sales'leri dogrusal regresyon grafik

    #Statsmodels ile modelleme
import statsmodels.api as sm
x = df[["TV"]]
x = sm.add_constant(x) #Hepsinin soluna 1 ekler
y = df["sales"] #Bagimsiz degisken

lm = sm.OLS(y,x) #Model kurma
model = lm.fit() #Modeli fitleme
model.summary() #Analiz

    #Farklý Yöntem
import statsmodels.formula.api as smf
lm = smf.ols("sales ~ tv",df)
model = lm.fit()
model.summary()
print(model.summary().tables[1]) #Mode.sum 0 1 2 olarak 3'e ayrilir
model.params[0]

print("f_pvalue: ","%.4f" % model.f_pvalue)
print("fvalue: ","%.2f" % model.fvalue)
print("tvalue: ","%.2f" % model.tvalues[0:1])
print("Sales = " + str("%.2f"% model.params[0]) +"TV" + "*" + str("%.2f" % model.params[1])) 
    

    #Tahmin
from sklearn.linear_model import LinearRegression
x = df[["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(x,y)
model.predict([[30]]) #33.Satirdeki  TV yerine gelcek sayi
yeni_veri = [[5],[90],[200]]
model.predict(yeni_veri)

    #Artiklarin onemi
from sklearn.metrics import mean_squared_error, r2_score
lm = smf.ols("sales ~ TV",df)
model = lm.fit()
model.summary()
mse = mean_squared_error(y,model.fittedvalues)
import numpy as np
rmse = np.sqrt(mse)
reg.predict(x)[0:10] #Tahmin edilenlen
y[0:10] #Gerçek deðerler
k_t = pd.DataFrame({"gercek": y[0:10],"tahmin":reg.predict(x)[0:10]})
k_t["hata"] = k_t["gercek"] - k_t["tahmin"]
k_t["hata_kare"] = k_t["hata"]**2
model.resid[0:10] #Modelin artýklarý
import matplotlib.pyplot as plt
plt.plot(model.resid)

    #Coklu Dogrusal Regresyon
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
x = df.drop("sales",axis=1)
y = df["sales"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42) #Test train ayri ayri tanimlanmasi daha iyi olur %20ye %80 olarak ayrildi
training = df.copy()

lm = sm.OLS(y_train,x_train)
model = lm.fit()
model.summary()

lm = LinearRegression()
model = lm.fit(x_train,y_train)
model.intercept_
model.coef_

    #Coklu dogrusal regresyon tahmin
#Sales = 2.97 + TV0.04 + radio0.18 + newspaper*0.002
#Ornegin 30 birim tv harcamasi, 10 birim radio, 40 birim gazete harcamasi oldugunda satislari tahmin deðeri ne olur
yeni_veri = [[30],[10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T
model.predict(yeni_veri) #Sonuc geldi
rmse = np.sqrt(mean_squared_error(y_train,model.predict(x_train)))
rmse #Egitim hatamiz
rmse = np.sqrt(mean_squared_error(y_test,model.predict(x_test)))
rmse #Test hatamiz

    #Model Dogrulama
lm = LinearRegression()
model = lm.fit(x_train,y_train)    
cross_val_score(model,x_train,y_train,cv=10,scoring="r2").mean() #Model icin 10 tane farkli r karenin ortalamasi    
cross_val_score(model,x_test,y_test,cv=10,scoring="neg_mean_squared_error")
    
    #PCR - Temel bilesen regresyonu
hit = pd.read_csv("Hitters.csv")
df = hit.copy()    
df = df.dropna()
df.head()
y = df["Salary"]
dms = pd.get_dummies(df[["League","Division","NewLeague"]])
x_ = df.drop(["Salary","League","Division","NewLeague"],axis = 1).astype("float64","int")    
x = pd.concat([x_, dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
pca = PCA()
x_reduced_train = pca.fit_transform(scale(x_train))    
x_reduced_train[0:1,:]
np.cumsum(np.round(pca.explained_variance_ratio_,decimals = 4)*100)[0:10] #% kacini acikliyoruz listesi 10. degiskende %97
lm = LinearRegression()
pcr_model = lm.fit(x_reduced_train,y_train)    
pcr_model.intercept_
pcr_model.coef_    

    #Tahmin
y_pred = pcr_model.predict(x_reduced_train)
y_pred[0:10]
np.sqrt(mean_squared_error(y_train, y_pred))
r2_score(y_train,y_pred)

pca2= PCA()
x_reduced_test = pca2.fit_transform(scale(x_test))
y_pred = pcr_model.predict(x_reduced_test)
np.sqrt(mean_squared_error(y_test,y_pred))

    #Model tuning ?
lm = LinearRegression()
pcr_model = lm.fit(x_reduced_train[:,0:2],y_train)    
y_pred = pcr_model.predict(x_reduced_test[:,0:2])
print(np.sqrt(mean_squared_error(y_test,y_pred)))
from sklearn import model_selection
cv_10 = model_selection.KFold(n_splits = 10,shuffle = True,random_state=1)
lm = LinearRegression()
RMSE = ()

    #PLS Kismi en kucuk kareler regresyonu
hit = pd.read_csv("Hitters.csv")
df = hit.copy()    
df = df.dropna()
y = df["Salary"]
dms = pd.get_dummies(df[["League","Division","NewLeague"]])
x_ = df.drop(["Salary","League","Division","NewLeague"],axis = 1).astype("float64","int")    
x = pd.concat([x_, dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
from sklearn.cross_decomposition import PLSRegression, PLSSVD
pls_model = PLSRegression(n_components=2).fit(x_train,y_train)
pls_model.coef_

    #PLS Tahmin
y_pred = pls_model.predict(x_train)
np.sqrt(mean_squared_error(y_train,y_pred))

y_pred = pls_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

    #PLS Model tuning
cv_10 = model_selection.KFold(n_splits = 10,shuffle = True,random_state=1)
RMSE = []

for i in np.arange(1,x_train.shape[1]+1):
    pls=PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls,x_train,y_train,cv=cv_10))
    RMSE.append(score)
import matplotlib.pyplot as plt
plt.plot(np.arange(1,x_train.shape[1]+1),np.array(RMSE),"-v",c="r")
plt.xlabel("Bilesen Sayisi")
plt.ylabel("RMSE")
plt.title("Salary")

    #Ridge Regresyon
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=0.1).fit(x_train,y_train)

lambdalar = 10**np.linspace(10,-2,100)*0.5

ridge_model = Ridge()
katsayilar = []
for i in lambdalar:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(x_train,y_train)
    katsayilar.append(ridge_model.coef_)

ax = plt.gca()
ax.plot(lambdalar,katsayilar)
ax.set_xscale("log")
plt.xlabel("Lambda(alpha) değerleri")
plt.ylabel("Katsayılar/Ağırlıklar")
plt.title("Düzenleştirmenin bir fonksiyonu olarak ridge katsayıları")

    #Ridge Tahmin
y_pred = ridge_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

    #Model tuning
lambdalar = 10**np.linspace(10,-2,100)*0.5
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas = lambdalar,scoring="neg_mean_squared_error",normalize = True)
ridge_cv.fit(x_train,y_train)
ridge_cv.alpha_
ridge_tuned = Ridge(alpha = ridge_cv.alpha_,normalize=True).fit(x_train,y_train)
np.sqrt(mean_squared_error(y_test,ridge_tuned.predict(x_test)))

    #Lasso Regresyon
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1).fit(x_train,y_train)
lasso_model.coef_
