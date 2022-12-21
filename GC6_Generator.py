import streamlit as st
import pandas as pd
import numpy as np
import pickle
#import matplotlib.pyplot as plt
#import xgboost as xgb


#from sklearn.metrics import accuracy_score
#from sklearn.metrics import r2_score,mean_squared_error
#from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


st.write("""
## Generator Health Index Prediction App
""")

st.image('./GT3301.png')

st.sidebar.header('Input Parameter for Simulation')

#st.sidebar.markdown("""
#[CSV input file]
#""")

# Collects user input features into dataframe

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.sidebar.selectbox('Select Model',['XGBoost Regression',
'Random Forest Regressor'])

if uploaded_file is not None:
    input_df2 = pd.read_csv(uploaded_file)
    input_df = input_df2.drop(columns=['Start_time','End_time','Severity'])
else:
    def user_input_features():
        Gen_DE = st.sidebar.slider('Temp.DE:33TIA427', 0, 220, 0)
        Gen_DE_1 = st.sidebar.slider('Temp.DE(1):33TIA428', 0, 220, 0)
        Cooling_ColdSide = st.sidebar.slider('Temp.Cooling_ColdSid:33TIA429', 0, 220, 0)
        Cooling_ColdSide_1 = st.sidebar.slider('Temp.Cooling_ColdSid(1):33TIA430', 0, 220, 0)
        Cooling_WarmSide = st.sidebar.slider('Temp.Cooling_WarmSide:33TIA431', 0, 220, 0)
        Cooling_WarmSide_1 = st.sidebar.slider('Temp.Cooling_WarmSide(1):33TIA432', 0, 220, 0)
        Cooling_ColdSide_2 = st.sidebar.slider('Temp.Cooling_WarmSide(2):33TIA433', 0, 220, 0)
        Cooling_ColdSide_3 = st.sidebar.slider('Temp.Cooling_WarmSide(3):33TIA434', 0, 220, 0)
        Gen_NDE = st.sidebar.slider('Temp.NDE:33TIA435', 0, 220, 0)
        Gen_NDE_1 = st.sidebar.slider('Temp.NDE(1):33TIA436', 0, 220, 0)
        Cooling_WarmSide_Exc = st.sidebar.slider('Temp.Cooling_WarmSide_Exc:33TIA437', 0, 220, 0)
        Gen_stator1 = st.sidebar.slider('Temp.stator1:33TIA438', 0, 220, 0)
        Gen_stator1_1 = st.sidebar.slider('Temp.stator1(1):33TIA439', 0, 220, 0)
        Gen_stator2 = st.sidebar.slider('Temp.stator2:33TIA440', 0, 220, 0)
        Gen_stator2_1 = st.sidebar.slider('Temp.stator2(1):33TIA441', 0, 220, 0)
        Gen_stator3 = st.sidebar.slider('Temp.stator3:33TIA442', 0, 220, 0)
        Gen_stator3_1 = st.sidebar.slider('Temp.stator3(1):33TIA443', 0, 220, 0)
        Gen_Ambiant = st.sidebar.slider('Temp.Ambient:33TIA445', 0, 220, 0)
        Gen_LubeOil = st.sidebar.slider('Temp.LubeOil:33-TIA-412', 0, 220, 0)
        Vi_XDE = st.sidebar.slider('RADIAL BEARING VIBRATION X D.E.', 0, 150,0)
        Vi_YDE = st.sidebar.slider('RADIAL BEARING VIBRATION Y D.E.', 0, 150,0)
        Vi_XNDE = st.sidebar.slider('RADIAL BEARING VIBRATION X N.D.E.', 0, 150,0)
        Vi_YNDE = st.sidebar.slider('RADIAL BEARING VIBRATION Y N.D.E.', 0, 150,0)
        Voltage = st.sidebar.slider('G 3301 Voltage', 0,20,0)
        Frequency = st.sidebar.slider('G 3301 Frequency', 0,60,0)
        PD_Gen_PhaseU = st.sidebar.slider('PD_Gen_PhaseU', 0 ,3000, 0)
        PD_Gen_PhaseV = st.sidebar.slider('PD_Gen_PhaseV', 0 ,3000, 0)
        PD_Gen_PhaseW = st.sidebar.slider('PD_Gen_PhaseW', 0 ,3000, 0)
        PD_Incoming = st.sidebar.slider('PD_Incoming', 0 ,150, 0)
        data = {
                'Temp.DE:33TIA427':Gen_DE,
                'Temp.DE(1):33TIA428':Gen_DE_1,
                'Temp.Cooling_ColdSid:33TIA429':Cooling_ColdSide,
                'Temp.Cooling_ColdSid(1):33TIA430':Cooling_ColdSide_1,
                'Temp.Cooling_WarmSide:33TIA431':Cooling_WarmSide,
                'Temp.Cooling_WarmSide(1):33TIA432':Cooling_WarmSide_1,
                'Temp.Cooling_WarmSide(2):33TIA433':Cooling_ColdSide_2,
                'Temp.Cooling_WarmSide(3):33TIA434':Cooling_ColdSide_3,
                'Temp.NDE:33TIA435':Gen_NDE,
                'Temp.NDE(1):33TIA436':Gen_NDE_1,
                'Temp.Cooling_WarmSide_Exc:33TIA437':Cooling_WarmSide_Exc,
                'Temp.stator1:33TIA438':Gen_stator1,
                'Temp.stator1(1):33TIA439':Gen_stator1_1,
                'Temp.stator2:33TIA440':Gen_stator2,
                'Temp.stator2(1):33TIA441':Gen_stator2_1,
                'Temp.stator3:33TIA442':Gen_stator3,
                'Temp.stator3(1):33TIA443':Gen_stator3_1,
                'Temp.Ambient:33TIA445':Gen_Ambiant,
                'Temp.LubeOil:33-TIA-412':Gen_LubeOil,
                'RADIAL BEARING VIBRATION X D.E.':Vi_XDE,
                'RADIAL BEARING VIBRATION Y D.E.':Vi_YDE,
                'RADIAL BEARING VIBRATION X N.D.E.':Vi_XNDE,
                'RADIAL BEARING VIBRATION Y N.D.E.':Vi_YNDE,
                'G 3301 Voltage':Voltage,
                'G 3301 Frequency':Frequency,
                'PD_Gen_PhaseU':PD_Gen_PhaseU,
                'PD_Gen_PhaseV':PD_Gen_PhaseV,
                'PD_Gen_PhaseW':PD_Gen_PhaseW,
                'PD_Incoming':PD_Incoming

                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire dataset

#scaler = StandardScaler()
#input_df = scaler.fit_transform(input_df)

df = pd.concat([input_df],axis=0)

# Selects only the 20 row (the user input data)
df = df[:20] 

# Displays the user input features
st.subheader('Parameter for Simulation')

if uploaded_file is not None:
   st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('20221201_GC6_Generator.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)

#----------------------------------------------------------

st.subheader('Simulation and Prediction')
#st.write([prediction])

#-----------------------------------------------------------
st.write('Severity')
st.write(prediction)
