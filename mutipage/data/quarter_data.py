import pandas as pd
import numpy as np
import streamlit as st

dataset = st.container()


with dataset:
    quarter_data = pd.read_csv("C:/Users/yakul/mutipage/data/quarter.csv")
    quarter_data.tail()
