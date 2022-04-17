import pandas as pd
import numpy as np
import streamlit as st

def create_quartertable():
    create_quartertable = pd.read_csv("C:/Users/yakul/mutipage/data/week.csv")
    create_quartertable.tail()
    return create_quartertable
