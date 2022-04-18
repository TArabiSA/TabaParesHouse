import pandas as pd
import numpy as np
import streamlit as st



def create_weektable():
    create_weektable = pd.read_csv("data/week.csv")
    create_weektable.tail()
    return create_weektable
