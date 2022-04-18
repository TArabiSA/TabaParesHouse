import pandas as pd
import numpy as np
import streamlit as st



def create_weektable():
    if uploaded_file is not None:
        df=pd.read_csv(“week.csv”)
        st.write(df)
   

   
