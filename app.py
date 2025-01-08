import pandas as pd 
import numpy as np  
import pickle as pk 
import streamlit as st

# Load the model
model = pk.load(open('model.pkl', 'rb'))   

st.header('Car Prediction ML App')


#importing the data
cars_data = pd.read_csv('used_cars.csv')
cars_data.dropna(inplace=True)


# create user input based on the dataset

#name of car
brand = st.selectbox('Select Car Brand', cars_data['brand'].unique())

# year 
model_year = st.slider('Production Year', min_value=1996, max_value=2024)

# mileage
mileage = st.slider('Mileage', min_value=0, max_value=200000)

#fuel type
fuel_type = st.selectbox('Fuel Type', cars_data['fuel_type'].unique())

# horsepower
engine = st.slider('Horsepower', min_value=0, max_value=800)

# accident
accident = st.selectbox('Accident?', cars_data['accident'].unique())

if st.button("Calculate"):
    input_data_model = pd.DataFrame({'brand': [brand], 'model_year': [model_year], 'mileage': [mileage], 'fuel_type': [fuel_type], 'engine': [engine], 'accident': [accident]})

    input_data_model['brand'].replace(['Ford', 'Hyundai', 'INFINITI', 'Audi', 'BMW', 'Lexus', 'Aston', 'Toyota',
    'Lincoln', 'Land', 'Mercedes-Benz', 'Dodge', 'Nissan', 'Jaguar', 'Chevrolet',
    'Kia', 'Jeep', 'Bentley', 'MINI', 'Porsche', 'Hummer', 'Chrysler', 'Acura',
    'Volvo', 'Cadillac', 'Maserati', 'Genesis', 'Volkswagen', 'GMC', 'RAM', 'Subaru',
    'Alfa', 'Ferrari', 'Scion', 'Mitsubishi', 'Mazda', 'Saturn', 'Honda', 'Bugatti',
    'Lamborghini', 'Rolls-Royce', 'McLaren', 'Buick', 'Lotus', 'Pontiac', 'FIAT',
    'Saab', 'Mercury', 'Plymouth', 'smart', 'Maybach', 'Suzuki'],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52], inplace=True)

    input_data_model.fuel_type.replace(['Gasoline', 'Hybrid', 'E85 Flex Fuel', 'Diesel', 'Plug-In Hybrid'], [1,2,3,4,5], inplace=True)

    input_data_model.accident.replace(['None reported', 'At least 1 accident or damage reported'], [1,2], inplace=True)
    # st.write(input_data_model)

    price = model.predict(input_data_model)
    if price > 0:
        st.write(f'Predicted Price: ${format(price[0], ".2f")}')
    else:
        st.write('WORTHLESS!!')
        st.write()
        st.write('A negative value was returned -- I apologize, the model is not perfect :/')
    

