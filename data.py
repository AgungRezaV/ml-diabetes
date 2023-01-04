import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt



def run_data():
    df = pd.read_csv('./dataset/diabetes.csv')

    st.subheader("Overwiew Data")
    st.dataframe(df)
    st.write(" Outcome : 0 untuk Negative")
    st.write(" Outcome : 1 untuk Positive")

    df['group'] = pd.cut(df['Age'], bins = [18, 30, 50, 99], include_lowest = True, labels = ['18-30', '30-50', '50-99'])
    df_age = df.groupby(by = 'group').mean()

    # fig, ax = plt.subplots()
    # ax.bar(x = df.index, height = df['Outcome'])
    # plot_age = plt.show()

    with st.expander("Data Types"):
        st.dataframe(df.dtypes)

    with st.expander("Summary"):
        st.dataframe(df.describe())

    with st.expander("Age"):
        st.dataframe(df_age)

    with st.expander("Outcome Distribution"):
        st.dataframe(df["Outcome"].value_counts().rename({0: "Negative", 1: "Positive"}, axis ='index'))