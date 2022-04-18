import pandas as pd
import numpy as np
import streamlit as st


def create_weektable():
    create_weektable = pd.read_csv("C:/Users/yakul/mutipage/data/week.csv")
    st.line_chart(create_weektable)
   

   
