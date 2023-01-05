import time
import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

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

    # Corr Matrix
    st.write("### Correlation Matrix")
    corr_matrix = myData.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(corr_matrix,
                    annot = True,
                    linewidths = 0.5,
                    fmt = ".2f",
                    cmap = "YlGnBu")
    bottom, top = ax.get_ylim()
    pr = ax.set_ylim(bottom + 0.5, top - 0.5)
    with st.expander("Correlation Matrix"):
        st.write(fig)
        st.write(pr)

    # Memisahkan Label Dan Fitur 
    X_pp = df.drop(['Outcome'], axis=1)
    X = preprocessing.scale(X_pp)
    scaler = preprocessing.StandardScaler().fit(X_pp)
    y = df['Outcome']

    input_data1, input_data2 = st.columns(2, gap="large")
    input_data1.subheader("Input Data X")
    input_data1.write(X)
    input_data2.subheader("Input Data y")
    input_data2.write(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    # Split data 
    st.write("## Split Data dengan SKlearn Model Selection")
    X_train, X_test, y_train, y_test  = train_test_split(X, y, shuffle = True, test_size=0.3, random_state=0, stratify = y)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Data X_train")
        st.write(X_train.shape)
    
    with col2:
        st.subheader("Data X_test")
        st.write(X_test.shape)
    
    with col3:
        st.subheader("Data y_train")
        st.write(y_train.shape)
    
    with col4:
        st.subheader("Data y_test")
        st.write(y_test.shape)

    # Model
    # knn = KNeighborsClassifier()
    # leaf_size = list(range(1,50))
    # n_neighbors = list(range(1,30))
    # p=[1,2]

    # hyperK = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    # gridK = GridSearchCV(knn, hyperK, scoring='roc_auc', 
    #                     cv = 5, verbose = 1, 
    #                     n_jobs = -1,)

    # bestK = gridK.fit(X_train, y_train)
    # The best parameters across ALL searched params:
    # {'leaf_size': 1, 'n_neighbors': 24, 'p': 1}

    #Modeling KNN
    st.write("### Modeling KNN")
    code_nb = '''knn = KNeighborsClassifier(leaf_size = 1,
                           n_neighbors = 24,
                           p = 1)'''
    st.code(code_nb, language='python')
    knn = KNeighborsClassifier(leaf_size = 1,
                           n_neighbors = 24,
                           p = 1)

    # Fit X_train dengan y_train
    knn.fit(X_train, y_train)
    st.write("### Fit data")
    code_fit = '''knn.fit(X_train, y_train))'''
    st.code(code_fit, language='python')

    # assign variabel Predict
    y_pred_knn = knn.predict(X_test)
    st.write("## Prediksi 20% data")
    st.write(y_pred_knn)

    # Melihat Confussion Matrix
    cm_knn = metrics.confusion_matrix(y_test, y_pred_knn)
    st.write("### Melihat Confussion Matrix")
    # code_cm = '''cm_knn'''
    # st.code(code_cm, language='python')
    st.code(cm_knn, language = 'python')

    # Plot Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm_knn, center=1, annot=True, fmt='g', ax=ax, linewidth=.8)
    ax.xaxis.tick_top()
    # Label Confusion Matrix
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Actual Value')
    ax.set_ylabel('Predicted Value')
    ax.xaxis.set_ticklabels(['Negatif', 'Positif'])
    ax.yaxis.set_ticklabels(['Negatif', 'Positif'])
    st.write(fig)

    # Perhitungan Accuracy, Precision, Recall model K-Nearest Neighbor
    acc_knn = metrics.accuracy_score(y_test, y_pred_knn)
    prec_knn = metrics.precision_score(y_test, y_pred_knn)
    rec_knn = metrics.recall_score(y_test, y_pred_knn)

    col_accuracy, col_precision, col_recall = st.columns(3, gap = "medium")
    col_accuracy.subheader("Accuracy")
    col_accuracy.write(acc_knn.round(2))
    col_precision.subheader("Precision")
    col_precision.write(prec_knn.round(2))
    col_recall.subheader("Recall")
    col_recall.write(rec_knn.round(2))

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
        prediction = knn.predict(scaledData)
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