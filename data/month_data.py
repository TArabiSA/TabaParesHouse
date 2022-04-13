import pandas as pd
import numpy as np
import streamlit as st

dataset = st.container()


with dataset:
    month_data = pd.read_csv("C:/Users/yakul/mutipage/data/month.csv")
    month_data.tail()
