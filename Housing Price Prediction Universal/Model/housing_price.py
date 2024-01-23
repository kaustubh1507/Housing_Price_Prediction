#Housing Price Prediction Model

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import json

matplotlib.rcParams["figure.figsize"]= (20,10)

df = pd.read_csv('bengaluru_house_prices.csv')
print(df.head())

print(df.groupby('area_type')['area_type'].agg('count')) # Groups the data as per elements in area_type and gives their total count

df1 = df.drop(['area_type','society','balcony','availability'], axis = 'columns') # Drops the 4 columns
print(df1.head())

print(df1.isna().sum())

df1.dropna(inplace = True) #Drops the rows with value NA
print(df1.isna().sum())

print(df1['size'].unique()) #Displays unique values in column size
df1['bhk'] = df1['size'].apply(lambda x: float(x.split(' ')[0])) # Create a new column bhk storing the 1 element of the string in size when tokenized
print(df1.head())

print(df1.total_sqft.unique())

def isFloat(x): #If x is a numerical value and can be converted to float it returns True. Else if x is not numerical (or is a range) it gives and exception thu returning False
    try:
        float(x)
    except:
        return False
    return True

print(df1[~df1['total_sqft'].apply(isFloat)]) #Applies isFloat on each row's total_sqft element and returns ones giving False values

def conv_sqft_to_num(x): #Converts range to avg of the 2, returns floats as they are and returns null in case of random values
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df2 = df1.copy()
df2['total_sqft'] = df2['total_sqft'].apply(conv_sqft_to_num) #Applies fn conv_sqft_to_num to total_sqft of each row
print(df2.head())

df3 = df2.copy()
df3['price_per_sqft'] = df3['price'] *100000 / df3['total_sqft']
print(df3.head())

df3.location = df3.location.apply(lambda x: x.strip()) #Removes blank space at end of location name
location_stats = df3.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)

print(len(location_stats[location_stats<=10]))
location_stats_less_than_10 = location_stats[location_stats<=10] #Locations with less than 10 occurences

df3.location = df3.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x) # Locn with less than 10 occurences classified as other
print(len(df3.location.unique()))

print(df3[df3.total_sqft/df3.bhk < 300])

df4 = df3[~(df3.total_sqft/df3.bhk < 300)]
print(len(df4))

print(df4.price_per_sqft.describe())

def remove_pps_outliers(df): #Removing outliers wrt price per sqft
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft> (m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df5 = remove_pps_outliers(df4)
print(len(df5))

def remove_bhk_outliers(df): #Removes flats with less bhk but more price than greater bhk flats
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df6 = remove_bhk_outliers(df5)

print(len(df6))

df7 = df6[df6.bath<df6.bhk+2]
print(df7.shape)

df8 = df7.drop(['size', 'price_per_sqft'],axis='columns')
print(df8.head())

dummies =  pd.get_dummies(df8.location)
print(dummies.head())

df9 = pd.concat([df8, dummies.drop('other', axis='columns')], axis='columns') #Create dummy variables for location
# print(df9.head())
df9.drop('location',axis = 'columns', inplace= True)
print(df9.head())
print(df9.shape)

X = df9.drop('price', axis= 'columns')
print(X.head())
Y = df9.price

xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.2, random_state=10) #Splitting dataset into training and test dataset
model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest,ytest))

def predict_price(location,sqft,bath,bhk): #Predicts price on the basis of given values
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if(loc_index)>=0:
        x[loc_index] = 1
    return model.predict([x])[0]

print(predict_price('1st Phase JP Nagar',1000,2,2))

with open ('bangalore_home_prices_model.pickle','wb') as f: # Storing model in pickle file
    pickle.dump(model,f)

columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))