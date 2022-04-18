import pandas as pd
import numpy as np
import streamlit as st
import csv
from data import week.csv

st.title('data')
file = pd.read_csv("C:/Users/yakul/mutipage/data/week_data.csv")
st.write(file)
   

   
