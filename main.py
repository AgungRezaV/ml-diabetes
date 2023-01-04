import streamlit as st
import streamlit.components.v1 as stc

from data import run_data
from ml import run_ml

html_temp = """
	<div style="background-color:#0A2647;padding:10px;border-radius:10px">
	<h1 style="color:#2C74B3;text-align:center;">Diabetes Health Risk Prediction Web</h1>
	<h4 style="color:white;text-align:center;">Kelompok 3</h4>
	</div>
    """

def main():
    #st.title("Main App")
    stc.html(html_temp)

    menu = ["Home", "Data", "ML"]
    choice = st.selectbox("Menu", menu)
    
    if choice == "Home":
        st.write("""
			### Early Stage Diabetes Risk Predictor App
			    Dataset ini berisikan data-data dari tanda dan gejala atas pasien diabetes baru atau yang akan menjadi pasien diabetes.
			#### Datasource
				- diabetes.csv
			#### Web Content
				- ML Section: ML Predictor Web
			""")
    elif choice == "Data":
        run_data()
    else:
        run_ml()

main()