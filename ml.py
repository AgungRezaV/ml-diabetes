import time
import streamlit as st 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB

# from help import 
from help import HelpFunction, help_glucose

gender_map = {"Female":0,"Male":1}

def run_ml():

    st.write("""
    # Program Prediksi Peluang Penyakit Diabetes
    """)

    st.write( """
        ## Keterangan Data Yang Digunakan
    """)

    st.write("""
        ### Overview Data
    """)

    myData = pd.read_csv('./dataset/diabetes.csv')

    st.dataframe(myData)

    st.write("""
        ### Deskripsi Data
    """)

    st.dataframe(myData.describe())

    # Preproccessing Data
    df = myData.copy()
    df[['Glucose','BloodPressure','SkinThickness','BMI']] = myData[['Glucose','BloodPressure','SkinThickness','BMI']].replace(0,np.NaN)

    def median_target(var):   
        temp = df[df[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        return temp

    median_target('Glucose')
    df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 107
    df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 140

    median_target('BloodPressure')
    df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70
    df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 74.5

    median_target('SkinThickness')
    df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27
    df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 32

    median_target('BMI')
    df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.1
    df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 34.3

    # Memisahkan Label Dan Fitur 
    X_pp = df.drop(['Outcome'], axis=1)
    X = preprocessing.scale(X_pp)
    scaler = preprocessing.StandardScaler().fit(X_pp)
    y = df['Outcome']

    st.write("### Input Data X",X)
    st.write("### Label Data y",y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    # Split data 
    st.write("## Split Data dengan SKlearn Model Selection")
    X_train, X_test, y_train, y_test  = train_test_split(X, y, shuffle = True, test_size=0.2, random_state=0, stratify = y)
    st.write("### Data X_train",X_train)
    st.write("### Data y_train",y_train)

    #Import Function Gaussian NB dari SKlearn naive_bayes
    st.write("### Import Function Gaussian NB dari SKlearn naive_bayes dan assign kedalam variabel")
    code_nb = '''gnb = GaussianNB()'''
    st.code(code_nb, language='python')
    gnb = GaussianNB()

    # Fit X_train dengan y_train
    gnb.fit(X_train, y_train)
    st.write("### Fit data")
    code_fit = '''gnb.fit(X_train, y_train)'''
    st.code(code_fit, language='python')
    # assign variabel Predict
    y_pred_gnb = gnb.predict(X_test)

    # Melihat Confussion Matrix
    cm_gnb = metrics.confusion_matrix(y_test, y_pred_gnb)
    st.write("### Melihat Confussion Matrix")
    code_cm = '''cm_gnb'''
    st.code(code_cm, language='python')
    st.code(cm_gnb, language = 'python')
    # Perhitungan Accuracy, Precision, Recall model Naive Bayes
    acc_gnb = metrics.accuracy_score(y_test, y_pred_gnb)
    prec_gnb = metrics.precision_score(y_test, y_pred_gnb)
    rec_gnb = metrics.recall_score(y_test, y_pred_gnb)

    st.write("Maka Dengan Menggunakan Naive Bayes Diperoleh Skor Accuracy")
    st.write(acc_gnb)
    st.write("Maka Dengan Menggunakan Naive Bayes Diperoleh Skor Precision")
    st.write(prec_gnb)
    st.write("Maka Dengan Menggunakan Naive Bayes Diperoleh Skor Recall")
    st.write(rec_gnb)

    st.write("## Sekarang Silahkan Masukan Data Untuk Mengetahui Prediksi Peluang Apakah Kamu Positif Atau Negatif Diabetes")

    with st.expander("Input Data"):
        with st.form("my_form"):
                # st.write("""
                #         Gula darah puasa (setelah tidak makan selama 8 jam): 70-99 mg/dL.
                #         Satu sampai dua jam setelah makan: kurang dari 140 mg/dL.
                #         Gula darah sewaktu: kurang dari 200 mg/dL.
                #         Gula darah sebelum tidur: 100-140 mg/dL.
                # """)
            inputPregnancies = st.number_input("Masukan Pregnancies Score: ", 0, help = "Jumlah Berapa Kali Hamil (Jika Laki-laki maka 0).")
            inputGlucose = st.number_input("Masukan Skor Glukosa: ", 0, help = HelpFunction(help_glucose))
            inputBP = st.number_input("Masukan Skor Tekanan Darah: ", 0, help = "Normal (tidak menderita diabetes): di bawah 120 mmHg.")
            inputST = st.number_input("Masukan Skor Ketebalan Kulit: ", 0, help = "Ketebalan lipatan kulit trisep (mm).")
            inputInsulin = st.number_input("Masukan Insulin: ", 0, help = "Insulin serum 2-jam (mu U/ml).")
            inputBMI = st.number_input("Masukan BMI: ", help="Body mass index (kg/(m)2).")
            inputBPF = st.number_input("Masukkan Diabetes Pedigree Function: ",help="Diabetes pedigree function (fungsi yang menilai kemungkinan diabetes berdasarkan riwayat keluarga). Contoh: 0.627") 
            inputAge = st.number_input("Umur :", 0, help="Tahun")
            submit = st.form_submit_button("Submit")

    completeData = np.array([inputPregnancies, inputGlucose, inputBP, 
                            inputST, inputInsulin, inputBMI, inputBPF, inputAge]).reshape(1, -1)
    scaledData = scaler.transform(completeData)

    st.write('Tekan Submit Untuk Melihat Prediksi Peluang Terkena Diabetes Anda')

    if submit : 
        prediction = gnb.predict(scaledData)
        with st.spinner('Wait for it...'):
            time.sleep(1)
            with st.expander("Prediction Results"):
                if prediction == 1 :
                    st.warning("## Anda Positif Diabetes") 
                    st.info("Silahkan konsultan ke rumah sakit terdekat.")
                else:
                    st.balloons()
                    # st.write(prediction)
                    st.success("## Anda Negatif Diabetes")


    # prediction = gnb.predict(scaledData)