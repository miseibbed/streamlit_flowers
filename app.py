#streamlit has a hosting service and can be deployed live

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pickle

#this creates an h1 header in our app
st.title("This is my flower predictor!")

#h2 header
st.header("This is a freaking cool app! :)")

#subheader
st.subheader("Flowers are so pretty")

#plotly has data built into it
df = px.data.iris()
# df   #streamlit shows the dataframe on the page

# display a boolean checkbox
show_df = st.checkbox("Do you want to see the data?")

# show_df is checked/true then show the df
if show_df:
    df

#create number input widgets
s_l = st.number_input("Sepal Length (cm)", 0, 100)
s_w = st.number_input("Sepal Width (cm)", 0, 100)
p_l = st.number_input("Petal Length (cm)", 0, 100)
p_w = st.number_input("Petal Width (cm)", 0, 100)

#what we want to do is take inputs, plug into model, and spit out preds
# this is 1d, so we need to reshape to 2d
user_input = np.array([s_l, s_w, p_l, p_w]).reshape(1,-1)

#read in pickl
with open('saved-iris-model.pkl', 'rb') as flower_pickle:
    model = pickle.load(flower_pickle)


#predict using the inputs
prediction = model.predict(user_input)
st.write(f'The predicted flower is {prediction[0]}!')

