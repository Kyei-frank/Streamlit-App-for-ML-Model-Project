# Importing required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing

# Setting up page configuration and directory path
st.set_page_config(page_title="Sales Forecasting App", page_icon="🐞", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Setting background image
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('images/background.jpg')

# Setting up logo
left1, left2, mid,right1, right2 = st.columns(5)
with mid:
    st.image("images/logo.jpg", use_column_width=True)

# Setting up Sidebar
social_acc = ['Data Field Description', 'EDA', 'About App']
social_acc_nav = st.sidebar.radio('**INFORMATION SECTION**', social_acc)

if social_acc_nav == 'Data Field Description':
    st.sidebar.markdown("<h2 style='text-align: center;'> Data Field Description </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown("**Date:** The date you want to predict sales  for")
    st.sidebar.markdown("**Family:** identifies the type of product sold")
    st.sidebar.markdown("**Onpromotion:** gives the total number of items in a product family that are being promoted at a store at a given date")
    st.sidebar.markdown("**Store Number:** identifies the store at which the products are sold")
    st.sidebar.markdown("**Holiday Locale:** provide information about the locale where holiday is celebrated")

elif social_acc_nav == 'EDA':
    st.sidebar.markdown("<h2 style='text-align: center;'> Exploratory Data Analysis </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''The exploratory data analysis of this project can be find in a Jupyter notebook from the linl below''')
    st.sidebar.markdown("[Open Notebook](https://github.com/Kyei-frank/Regression-Project-Store-Sales--Time-Series-Forecasting/blob/main/project_workflow.ipynb)")

elif social_acc_nav == 'About App':
    st.sidebar.markdown("<h2 style='text-align: center;'> Sales Forecasting App </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown("This App predicts the sales for product families sold at Favorita stores using regression model.")
    st.sidebar.markdown("")
    st.sidebar.markdown("[ Visit Github Repository for more information](https://github.com/Kyei-frank/Regression-Project-Store-Sales--Time-Series-Forecasting)")

# Loading Machine Learning Objects
@st.cache()
def load_saved_objects(file_path = 'ML_items'):
    # Function to load saved objects
    with open('ML_items', 'rb') as file:
        loaded_object = pickle.load(file)
        
    return loaded_object

# Instantiating ML_items
Loaded_object = load_saved_objects(file_path = 'ML_items')
model, encoder, train_data, stores, holidays_event = Loaded_object['model'], Loaded_object['encoder'], Loaded_object['train_data'], Loaded_object['stores'], Loaded_object['holidays_event']

# Setting Function for extracting Calendar features
@st.cache()
def getDateFeatures(df, date):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.isocalendar().week
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df['is_weekend']= np.where(df['day_of_week'] > 4, 1, 0)
    df['is_month_start']= df.date.dt.is_month_start.astype(int)
    df['is_month_end']= df.date.dt.is_month_end.astype(int)
    df['quarter']= df.date.dt.quarter
    df['is_quarter_start']= df.date.dt.is_quarter_start.astype(int)
    df['is_quarter_end']= df.date.dt.is_quarter_end.astype(int)
    df['is_year_start']= df.date.dt.is_year_start.astype(int)
    
    return df

# Setting up variables for input data
@st.cache()
def setup(tmp_df_file):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame(
        dict(
            date=[],
            store_nbr=[],
            family=[],
            onpromotion=[],
            city=[],
            state=[],
            store_type=[],
            cluster=[],
            day_type=[],
            locale=[],
            locale_name=[],
        )
    ).to_csv(tmp_df_file, index=False)

# Setting up a file to save our input data
tmp_df_file = os.path.join(DIRPATH, "tmp", "data.csv")
setup(tmp_df_file)

# setting Title for forms
st.markdown("<h2 style='text-align: center;'> Sales Prediction </h2> ", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center;'> Fill in the details below and click on SUBMIT button to make a prediction for a specific date and item </h7> ", unsafe_allow_html=True)

# Creating columns for for input data(forms)
left_col, mid_col, right_col = st.columns(3)

# Developing forms to collect input data
with st.form(key="information", clear_on_submit=True):
    
    # Setting up input data for 1st column
    left_col.markdown("**PRODUCT DATA**")
    date = left_col.date_input("Prediction Date:")
    family = left_col.selectbox("Item family:", options= list(train_data["family"].unique()))
    onpromotion = left_col.selectbox("Onpromotion code:", options= set(train_data["onpromotion"].unique()))
    store_nbr = left_col.selectbox("Store Number:", options= set(stores["store_nbr"].unique()))
    
    # Setting up input data for 2nd column
    mid_col.markdown("**STORE DATA**")
    city = mid_col.selectbox("City:", options= set(stores["city"].unique()))
    state = mid_col.selectbox("State:", options= list(stores["state"].unique()))
    cluster = mid_col.selectbox("Store Cluster:", options= list(stores["cluster"].unique()))
    store_type = mid_col.radio("Store Type:", options= set(stores["store_type"].unique()), horizontal = True)

    # Setting up input data for 3rd column
    right_col.markdown("**ADDITIONAL DATA**")
    check= right_col.checkbox("Is it a Holiday or weekend?")
    if check:
        right_col.write('Fill the following information on Day Type')
        day_type = right_col.selectbox("Holiday:", options= ('Holiday','Special Day:Transfered/Additional Holiday','No Work/Weekend'))
        locale= right_col.selectbox("Holiday Locale:", options= list(holidays_event["locale"].unique()))
        locale_name= right_col.selectbox("Locale Name:", options= list(holidays_event["locale_name"].unique()))
    else:
        day_type = 'Workday'
        locale = 'National'
        locale_name= 'Ecuador'
 
    submitted = st.form_submit_button(label="Submit")

# Setting up background operations after submitting forms
if submitted:
    # Saving input data as csv after submission
    pd.read_csv(tmp_df_file).append(
        dict(
                date = date,
                store_nbr = store_nbr,
                family=family,
                onpromotion= onpromotion,
                city=city,
                state=state,
                store_type=store_type,
                cluster=cluster,
                day_type=day_type,
                locale=locale,
                locale_name=locale_name
            ),
            ignore_index=True,
    ).to_csv(tmp_df_file, index=False)
    st.balloons()

    # Converting input data to a dataframe for prediction
    df = pd.read_csv(tmp_df_file)
    df= df.copy()
        
    # Getting date Features
    processed_data= getDateFeatures(df, 'date')
    processed_data= processed_data.drop(columns=['date'])
    
    # Encoding Categorical Variables
    encoder = preprocessing.LabelEncoder()
    cols = ['family', 'city', 'state', 'store_type', 'locale', 'locale_name', 'day_type']
    for col in cols:
        processed_data[col] = encoder.fit_transform(processed_data[col])
    
    # Making Predictions
    def predict(X, model):
        results = model.predict(X)
        return results
    
    prediction = predict(X= processed_data, model= Loaded_object['model'])
    df['Sales']= prediction 
    
    
    # Displaying prediction results
    st.markdown('''---''')
    st.markdown("<h4 style='text-align: center;'> Prediction Results </h4> ", unsafe_allow_html=True)
    st.success(f"Predicted Sales: {prediction[-1]}")
    st.markdown('''---''')

    # Making expander to view all records
    expander = st.expander("See all records")
    with expander:
        df = pd.read_csv(tmp_df_file)
        df['Sales']= prediction
        st.dataframe(df)
