import pandas as pd
import numpy as np
import streamlit as st
import week as data


def create_weektable():
    create_weektable = pd.read_csv("C:/Users/yakul/mutipage/data/week.csv")
    create_weektable.tail()
    return create_weektable
