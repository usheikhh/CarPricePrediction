import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing the data
cars_data = pd.read_csv('used_cars.csv')

# starting preprocessing

# print(cars_data.transmission.value_counts() - too many different values for transmission; will drop category
cars_data.drop(['transmission'], axis=1, inplace=True)

# print(cars_data.shape) (4409,11)

# checking for nulls
# print(cars_data.isnull().sum())- null values in fuel_type, accident, clean_title, 
cars_data.dropna(inplace=True)
# print(cars_data.shape) - dropped to (3269,11)

# Duplicates Check
# print(cars_data.duplicated().sum()) - 0 duplicates

# still at (3269,11)

# Check datatype for each col
# print(cars_data.info()) - everything except model_year is object

# Data Analysis 
# for col in cars_data.columns:
#     print("Unique values of",col)
#     print(cars_data[col].unique())
#     print("\n")

# only one value for clean_title - need to drop
# can modify unique values of engine to contain only the horsepower - can turn that into a numerical value
# can modify unique value values of price - change from string to numerical
# can modify unique values of mileage to contain only the numerical value


cars_data.drop(['clean_title'], axis=1, inplace=True)
cars_data.drop(['ext_col'], axis=1, inplace=True) # there are errors in the ds
cars_data.drop(['int_col'], axis=1, inplace=True) # there are errors in the ds
cars_data.drop(['model'], axis=1, inplace=True) # too many unique values does not make sense to change into numerical values


def get_hp(engine):
    if not 'HP' in engine: #some engine values only contain the amount of liters 
        return 'remove'
    
    engine = engine.split(' ')[0]
    engine = engine.strip()
    return (int)(engine[:-4]) #getting rid of '.0 HP' and converting to integer

cars_data['engine'] = cars_data['engine'].apply(get_hp) # applying the function to get numerical values 
cars_data = cars_data[cars_data['engine'] != 'remove'] # getting rid of values that did not contain horsepower

#converting the price to a numerical value
def get_price(price):
    price = price.replace('$', '')
    price = price.replace(',', '')
    return int(price)

cars_data['price'] = cars_data['price'].apply(get_price)

#converting the mileage to a numerical value
def get_miles(mileage):
    mileage = mileage.replace('mi.', '')
    mileage = mileage.replace(',', '')
    return int(mileage)

cars_data['mileage'] = cars_data['mileage'].apply(get_miles)

# replacing the values of the brand with numerical values
# print(cars_data['brand'].nunique())
cars_data['brand'].replace(['Ford', 'Hyundai', 'INFINITI', 'Audi', 'BMW', 'Lexus', 'Aston', 'Toyota',
 'Lincoln', 'Land', 'Mercedes-Benz', 'Dodge', 'Nissan', 'Jaguar', 'Chevrolet',
 'Kia', 'Jeep', 'Bentley', 'MINI', 'Porsche', 'Hummer', 'Chrysler', 'Acura',
 'Volvo', 'Cadillac', 'Maserati', 'Genesis', 'Volkswagen', 'GMC', 'RAM', 'Subaru',
 'Alfa', 'Ferrari', 'Scion', 'Mitsubishi', 'Mazda', 'Saturn', 'Honda', 'Bugatti',
 'Lamborghini', 'Rolls-Royce', 'McLaren', 'Buick', 'Lotus', 'Pontiac', 'FIAT',
 'Saab', 'Mercury', 'Plymouth', 'smart', 'Maybach', 'Suzuki'],
 [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52], inplace=True)


# print(cars_data.fuel_type.value_counts())
cars_data.fuel_type.replace(['Gasoline', 'Hybrid', 'E85 Flex Fuel', 'Diesel', 'Plug-In Hybrid'], [1,2,3,4,5], inplace=True)
# print(cars_data.accident.value_counts())    
cars_data.accident.replace(['None reported', 'At least 1 accident or damage reported'], [1,2], inplace=True)


# splitting input vs output columns 
input_data = cars_data.drop(['price'], axis=1)
output_data = cars_data['price']


# splitting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2) # 80% training, 20% testing

# MODEL CREATION  - linear regression 

model = LinearRegression(positive=True)

# training the model
model.fit(x_train, y_train)

predict = model.predict(x_test) # predicting the price of the cars
# print(predict)


# print(x_train.head(1))
#       brand  model_year  mileage  fuel_type engine  accident
# 3007      5        2016   185000          1    180         2

#           brand  model_year  mileage  fuel_type engine  accident
# 2482      8        2007   136000          1    273         1



input_data_model = pd.DataFrame({'brand': [5], 'model_year': [2016], 'mileage': [185000], 'fuel_type': [1], 'engine': [180], 'accident': [2]}) # should be 7500
# print(model.predict(input_data_model)) 
# orginally gave -20k, need to fix the model
# made coefs. positive=True, now it returns 3950




import pickle as pk
pk.dump(model, open('model.pkl', 'wb')) # saving the model
