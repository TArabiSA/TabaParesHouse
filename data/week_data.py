import pandas as pd
import numpy as np
import streamlit as st



def create_weektable():
    fig = pandas.read_csv("data/week.csv")
    fig.tail()
    return fig
   

   
