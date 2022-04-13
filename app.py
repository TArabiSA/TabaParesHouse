from PIL import Image
import streamlit as st
from multiapp import MultiApp
import week, month, quarter, year





st.set_page_config(page_title="Taba Pares House", page_icon=":chart_with_upwards_trend:", layout= "wide")
img_contact_form = Image.open("C:/Users/yakul/mutipage/images/tph.jpg")


app = MultiApp()
st.markdown("""
# Taba Pares House Sale's Forecasting using LSTM
[Fabook page](https://www.facebook.com/TABAPARESHOUSE).
""")

# Add all your application here
app.add_app("WEEK", week.app)
app.add_app("MONTH", month.app)
app.add_app("QUARTER", quarter.app)
app.add_app("YEAR", year.app)
# The main app
app.run()
