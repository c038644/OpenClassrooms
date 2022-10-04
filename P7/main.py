##########################################################
# to run: streamlit run main.py
##########################################################
from app import global_data, local
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
#import urllib.request

st.set_page_config(page_title='Credit Rating Calculator',  layout='wide', page_icon=':Calculator:')

#this is the header
 
t1, t2 = st.columns((0.07,1)) 

t2.title("Credit Rating Calculator")
t2.markdown("with Global and Local Customer Data")



with st.spinner('Updating Report...'):
    
    Customer_ID = pd.read_csv("files/Customer_ID.csv")
    Customer_ID = Customer_ID.drop(columns=['Unnamed: 0'])
    
    all_data = pd.read_csv("files/P7_test_df.csv")
    all_data = all_data.drop(columns=['Unnamed: 0'])

    Customer = st.selectbox('Select Customer', Customer_ID, help = 'Filter report to show only one customer')

    #API_location = "http://127.0.0.1:5000/local" #+ Customer_ID
    #json_url = urlopen(API_location)
    #API_data = json.loads(json_url.read())
    #st.API_data

    if Customer:
        Selected_Customer = all_data.loc[all_data['SK_ID_CURR'] == Customer]
        st.write(Selected_Customer)
        Selected_Customer.to_csv("files/selection.csv")
        local = requests.get("http://127.0.0.1:5000/local").json()
        #st.json(local) 
       

    g1, g2, g3 = st.columns((1,1,1))

    local_graph_df = pd.read_csv("files/Customer_score.csv")
    
    fig = px.bar(local_graph_df, x = 'Feature', y='Importance')
    
 
    fig.update_layout(title_text="Local Features Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
    
    g1.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = local_graph_df.iat[0,3],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Credit Rating", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.26], 'color': 'red'},
                {'range': [0.26, 0.36], 'color': 'orange'},
                {'range': [0.36, 1], 'color': 'green'}],
            'threshold': {
                'line': {'color': "blue", 'width': 4},
                'thickness': 0.75,
                'value': 0.31}}))

    fig2.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

    g2.plotly_chart(fig2, use_container_width=True) 

    dashboard = requests.get("http://127.0.0.1:5000/global_data").json()

    #global_data() 

    global_graph_df = pd.read_csv("files/Global_Features.csv")

    global_graph_df = global_graph_df.drop(columns=['Unnamed: 0'])
    
    fig = px.bar(global_graph_df, x = 'Feature', y='Importance')
    
    fig.update_layout(title_text="Global Features Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
    
    g3.plotly_chart(fig, use_container_width=True)

    Selected_Customer = pd.read_csv("files/selection.csv")

    Feature_List = pd.read_csv("files/P7_Features.csv")

    Feature = st.selectbox('Select Feature', Feature_List, help = 'Filter report to show only one feature')

    Selected_Feature = all_data.loc[all_data[Feature] == Feature].any()

    g4, g5 = st.columns((1,2))

    fig = px.scatter(Selected_Customer, x = 'SK_ID_CURR', y = Feature)
    
    fig.update_layout(title_text="Local Feature Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
    
    g4.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(all_data, x = 'SK_ID_CURR', y = Feature)
    
    fig.update_layout(title_text="Global Feature Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
    
    g5.plotly_chart(fig, use_container_width=True)


#*********************************************************

#st.write("Select a Customer")#

#def main_table(df: pd.DataFrame):

#    options = GridOptionsBuilder.from_dataframe(
#        df, enableRowGroup=True, enableValue=True, enablePivot=True
#    )

#    options.configure_side_bar()

#    options.configure_selection("single")
#    selection = AgGrid(
#        df,
#        enable_enterprise_modules=True,
#        gridOptions=options.build(),
#        theme="light",
#        update_mode=GridUpdateMode.MODEL_CHANGED,
#        allow_unsafe_jscode=True,
#    )

#    return selection

#Selected_Columns = pd.read_csv(
#    "C:/Users/Farida/Documents/Data_Science/P7/Final/files/Selected_Columns.csv"
#)

#selection = main_table(df=Selected_Columns)

#if selection:
#    st.write("You selected:")
#    st.json(selection["selected_rows"])
#    select_rows = json.dumps(selection["selected_rows"])
#    Customer = pd.read_json(select_rows)
#    Customer.to_csv(
#    "C:/Users/Farida/Documents/Data_Science/P7/Final/files/selection.csv"
#    )
    
#    local = requests.get("http://localhost:5000/local").json()
#    st.json(local)

#dashboard = requests.get("http://localhost:5000/dashboard").json()
#st.json(dashboard)


#selection1 = result_table(df=local)

#dashboard = requests.get("http://localhost:5000/global").json()

# labels
#labels = requests.get("http://localhost:5000/api/labels").json()
#selector = st.multiselect("Select Customer:", labels)

#load data
#data = pd.read_json(
#    requests.get("http://localhost:5000/api/data", params={"selector": selector}).json()
#)

# setup figure
#fig = px.scatter(
#    x=data["SK_ID_CURR"],
#    y=data["TARGET"],
#)
#st.write(fig)