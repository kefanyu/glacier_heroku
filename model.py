#%%
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

#%%

toy = pd.read_csv('toy.csv')
X = toy.loc[:, ['Company','KM Travelled','Cost of Trip']]
y = toy.loc[:,'Price Charged']

#%%

regressor = LinearRegression()
regressor.fit(X,y)

#%%

pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))