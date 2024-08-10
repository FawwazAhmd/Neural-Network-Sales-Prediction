import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\fnn09\Downloads\\train_v9rqX0R.xls")

df.head() 

df["Item_Identifier"].value_counts()
df.shape 

df["Item_Fat_Content"].unique()
df['Item_Type'].unique()
df['Outlet_Establishment_Year'].unique()


df['Outlet_Size'].isnull().sum()
df['Item_Weight'].isnull().sum()

#Dropping Reduntant Features 

df.drop(['Outlet_Identifier','Outlet_Establishment_Year'],axis = 1,inplace = True)
df.drop(['Item_Identifier'],axis = 1,inplace = True)
df.head()

df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].interpolate())

df.dtypes
df.info() 

#Mapping Categorical Columns

df['Item_Fat_Content']=df['Item_Fat_Content'].map({'Low Fat':0,'Regular':1,'LF':2,'reg':3,'low fat':4})

df['Item_Type']= df['Item_Type'].map({'Fruits and Vegetables':0,'Snack Foods':1,'Household':2,'Frozen Foods':3,'Dairy':4,'Canned':5,'Baking Goods':6,'Health and Hygiene':7,'Soft Drinks':8,'Meat':9,'Breads':10,'Hard Drinks':11,'Others':12,'Starchy Foods':13,'Breakfast':14,'Seafood':15})

df['Outlet_Size']=df['Outlet_Size'].map({'Medium':0,'Small':1,'High':2})

df['Outlet_Type']= df['Outlet_Type'].map({'Supermarket Type1':0,'Grocery Store':1,'Supermarket Type3':2,'Supermarket Type2':3})

df['Outlet_Location_Type']= df['Outlet_Location_Type'].map({'Tier 1':0,'Tier 2':1,'Tier 3':2})

df.head()

#Normalizing Data using Min Max Normalization

col = ["Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Type","Outlet_Location_Type","Item_Weight","Item_Outlet_Sales","Item_MRP"]
for i in col:
    df[i] = (df[i]-df[i].min())/(df[i].max()-df[i].min())

df.head()

#Training the model 

X =df.drop(['Item_Outlet_Sales'],axis=1)
Y =df['Item_Outlet_Sales']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size =0.2,random_state=10)

# shape of training and validation set
(x_train.shape, y_train.shape), (x_test.shape, y_test.shape)

#Implementing Neural Network 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

model = Sequential(
    [
        InputLayer(input_shape=(8,)),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(5, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='linear')
    ]
)

model.compile(loss='mse',optimizer='adam',metrics=['mse','mae'])

model_history= model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100)

#Model Prediction 

model_predictions=model.predict(x_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(y_test,model_predictions))
print(mean_squared_error(y_test,model_predictions))

from sklearn.metrics import r2_score
print("X test r2 score: ",r2_score(y_test,model_predictions))

model_predictions_train=model.predict(x_train)
print("X train e2 score: ",r2_score(y_train,model_predictions_train))

#Visualizing loss 

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show() 