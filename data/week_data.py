import pandas as pd
import numpy as np
import streamlit as st

dataset = st.container()


with dataset:
    week_data = pd.read_csv("C:/Users/yakul/mutipage/data/week.csv")
    week_data.tail()
