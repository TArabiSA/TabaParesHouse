import pandas as pd
import numpy as np
import streamlit as st


def create_yeartable():
    create_yeartable = pd.read_csv("C:/Users/yakul/mutipage/data/week.csv")
    create_yeartable.tail()
    return create_yeartable

