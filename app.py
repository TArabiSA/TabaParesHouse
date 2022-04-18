import streamlit as st
st.set_page_config(page_title="Taba Pares House", page_icon=":chart_with_upwards_trend:", layout= "wide")
from multiapp import MultiApp
from apps import week_app, month_app, quarter_app, year_app



app = MultiApp()
st.markdown("""
# Taba Pares House Sale's Forecasting using LSTM
[Fabook page](https://www.facebook.com/TABAPARESHOUSE).
""")

# Add all your application here
app.add_app("WEEK", week_app.app)
app.add_app("MONTH", month_app.app)
app.add_app("QUARTER", quarter_app.app)
app.add_app("YEAR", year_app.app)
# The main app
app.run()
