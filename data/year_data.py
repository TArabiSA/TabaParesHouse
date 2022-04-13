import pandas as pd
import numpy as np
import streamlit as st

dataset = st.container()


with dataset:
    year_data = pd.read_csv('C:/Users/yakul/mutipage/data/year.csv')
    year_data.tail()

