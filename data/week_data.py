import pandas as pd
import numpy as np
import streamlit as st



def create_monthtable():
    create_monthtable = pd.read_csv("data/week.csv")
    create_monthtable.tail()
    return create_monthtable
   

   
