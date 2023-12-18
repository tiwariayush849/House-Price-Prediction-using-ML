#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# In[5]:


data = pd.read_csv('Bengaluru_House_Data.csv')


# In[6]:


data.head()


# In[7]:


data.shape #13320 rows and 9 columns


# In[8]:


data.info()


# In[9]:


for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)


# In[10]:


data.isna().sum() # To know the number of null values of the columns in the datas


# In[11]:


data.drop(columns=['area_type','availability','society','balcony'],inplace=True) 


# In[12]:


data.describe() #price will be in lakhs


# In[13]:


data.info() #TO know which are the columns left


# In[14]:


data['location'].value_counts() # To know the missing values


# In[15]:


data['location'] = data['location'].fillna('Sarjapur Road') # We have fill the location 


# In[16]:


data['size'].value_counts() #For the 2nd column,size


# In[17]:


data['size']=data['size'].fillna('2 BHK') # we fill the 16 missing value of size with 2 BHK
data['bath'] = data['bath'].fillna(data['bath'].median()) #To remove the null values from the bathroom
data.info() #Now there is no null values


# In[18]:


data['bhk']=data['size'].str.split().str.get(0).astype(int)
data[data.bhk >20] #which are the flats whose bhk is greater than 20


# In[19]:


data['total_sqft'].unique() #since there is a problem of range,so we have to fix


# In[20]:


#Made a function to conert the range, by taking a mean
def convertRange(x):
    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0])+float(temp[1]))/2 #making float of mean of temp
    try: #excption handlaning
        return float(x)
    except:
        return None


# In[21]:


data['total_sqft'] = data['total_sqft'].apply(convertRange)


# In[22]:


data.head()


# In[23]:


#To removing outliers now i have made price per square feet,=(price/total_sqft) and will chace the column name as price_per_sqft
data['price_per_sqft']=data['price'] *100000 / data['total_sqft']


# In[24]:


data['price_per_sqft']


# In[25]:


data.describe()


# In[26]:


data['location'].value_counts()


# In[27]:


#now we will reduce the number of locations to as there are 1 value in many locations we will write it in 'other'
data['location']=data['location'].apply(lambda x:x.strip())
location_count = data['location'].value_counts()
location_count #This is my new location counts


# In[28]:


location_count_less_10 = location_count[location_count<=10] #To find out the number of locations which are lessthen 10
location_count_less_10


# In[29]:


data['location'] = data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)
#where the loaction will be less then 10 then it will not show the place,it will show as other
data['location'].value_counts()


# In[30]:


#outlier detection and removal
data.describe()


# In[31]:


#TO find out how many flats should preasent in 1 BHK
(data['total_sqft']/data['bhk']).describe()


# In[32]:


#We will keep where total_sqft/data_bhk>=300
data = data[(data['total_sqft']/data['bhk'])>=300]
data.describe()
#now the minimum has become 300


# In[33]:


data.shape


# In[34]:


data.price_per_sqft.describe()


# In[35]:


#To remove the outilers,i;e the max value we have to write a function
def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        
        st = np.std(subdf.price_per_sqft)
        
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output,gen_df],ignore_index =True)
    return df_output
data = remove_outliers_sqft(data)
data.describe()


# In[36]:


#To remove the outliers form 'bhk'
def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {} #dictionary
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean': np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
       # print(location,bhk_stats)
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


# In[37]:


data = bhk_outlier_remover(data)
data.shape 


# In[38]:


data


# In[39]:


data.drop(columns=['size','price_per_sqft'],inplace=True) #we have drop size and price_per_sqft
data.head() #Now the data looks like


# In[40]:


data.to_csv("Cleaned_data.csv") #saved the clean data
X=data.drop(columns=['price'])
y=data['price']
#Now importing all the predefined libreris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
#Now split the train,i;e x_train and y_train
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)


# In[41]:


#Now i have use Linear Regression
column_trans = make_column_transformer((OneHotEncoder(sparse=False),['location']),remainder = 'passthrough')
#we have used location column because it is the only column which has a string 
scaler = StandardScaler()


# In[42]:


lr = LinearRegression()


# In[43]:


pipe = make_pipeline(column_trans,scaler,lr)
pipe.fit(X_train,y_train)


# In[44]:


y_pred_lr = pipe.predict(X_test)
r2_score(y_test, y_pred_lr)


# In[45]:


#Now i apply Lasso Reg
lasso = Lasso()
pipe = make_pipeline(column_trans,scaler, lasso)
pipe.fit(X_train,y_train)


# In[46]:


y_pred_lasso = pipe.predict(X_test)
r2_score(y_test, y_pred_lasso)


# In[47]:


#Now i have use Ridge Reg
ridge = Ridge()
pipe = make_pipeline(column_trans,scaler, ridge)
pipe.fit(X_train,y_train)


# In[48]:


y_pred_ridge = pipe.predict(X_test)
r2_score(y_test, y_pred_ridge)


# In[49]:


print("No Regularization: ", r2_score(y_test, y_pred_lr))
print("Lasso: ", r2_score(y_test, y_pred_lasso))
print("Ridge:", r2_score(y_test, y_pred_ridge))


# In[50]:


import pickle


# In[52]:


filename = 'ridge_model.pkl'
pickle.dump(pipe, open(filename, 'wb'))


# In[54]:


loaded_model = pickle.load(open(filename, 'rb'))


# In[55]:


example_data = pd.DataFrame({
    'location': ['Whitefield'],
    'total_sqft': [1500],
    'bath': [3],
    'bhk': [3]
})


# In[56]:


predicted_price = loaded_model.predict(example_data)
print(f'Predicted Price: {predicted_price[0]}')


# In[ ]:




