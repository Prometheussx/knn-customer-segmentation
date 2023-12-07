#%% Library Imort
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

#%% Data İmport
df = pd.read_csv('teleCust1000T.csv')
df.head()

#%% Data Visualization and Data Analysis
df['custcat'].value_counts()
df.hist(column='income', bins=50)

# %% Feature Set
df.columns
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']].values
X[0:5] #Values Visualization

y = df['custcat'].values
y[0:5]
# %% Train and Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=96) 
print("Train set:", X_train.shape, y_train.shape)
print("Train set:", X_test.shape, y_test.shape)

#%% Normalize Data
X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_train_norm[0:5]
# %%Training Model
from sklearn.neighbors import KNeighborsClassifier
k = 4
neigh = KNeighborsClassifier(n_neighbors=k,).fit(X_train_norm,y_train)
neigh
#%% Test Normalize Data
X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
X_test_norm[0:5]
# %%Predicting
yhat = neigh.predict(X_test_norm)
print("Tahmin sonucu:",yhat[0:5])
print("Doğru Sonuç",y_test[0:5])

# %% Accuracy Evaluation
from sklearn import metrics
print("Train set Acc:", metrics.accuracy_score(y_train,neigh.predict(X_train_norm)))
print("Test set Acc", metrics.accuracy_score(y_test,yhat))
# %%
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train_norm,y_train)
    yhat=neigh.predict(X_test_norm)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print("mean_acc",mean_acc)
print("std_acc",std_acc)
# %% Plot Model Acc for K values
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
# %%Accuracy
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 