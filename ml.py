import streamlit as st 
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 

def run_ml():
    st.write("""
    # Program Prediksi Peluang Penyakit Diabetes
""")

st.write( """
    ## Keterangan Data Yang Digunakan
""")

st.write("""
    ## Overview Data
""")

myData = pd.read_csv('./dataset/diabetes.csv')

st.dataframe(myData)

st.write("""
    ## Deskripsi Data
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

# st.write("""
#     ## Dilakukan Preprocessing Data dimana Fitur dan Labelnya akan Dipisah

# """)
# st.dataframe()

# Memisahkan Label Dan Fitur 
X_pp = df.drop(['Outcome'], axis=1)
X = preprocessing.scale(X_pp)
scaler = preprocessing.StandardScaler().fit(X_pp)
y = df['Outcome']

# X = myData.iloc[:, 0:-1].values
# y = myData.iloc[:, -1].values

st.write("## Input Data X",X)
st.write("## Label Data y",y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
# Split data 
st.write("## Split Data dengan SKlearn Model Selection")
X_train, X_test, y_train, y_test  = train_test_split(X, y, shuffle = True, test_size=0.2, random_state=0, stratify = y)
st.write("## Data X_train",X_train)
st.write("## Data y_train",y_train)

#Import Function Gaussian NB dari SKlearn naive_bayes
st.write("## Import Function Gaussian NB dari SKlearn naive_bayes dan assign kedalam variabel")
st.write("gnb = GaussianNB()")
gnb = GaussianNB()

# Fit X_train dengan y_train
gnb.fit(X_train, y_train)
st.write("Fit data")
st.write("gnb.fit(X_train, y_train)")
# assign variabel Predict
y_pred_gnb = gnb.predict(X_test)

# Melihat Confussion Matrix
cm_gnb = metrics.confusion_matrix(y_test, y_pred_gnb)
st.write("Melihat Confussion Matrix")
cm_gnb

# Perhitungan Accuracy, Precision, Recall model Naive Bayes
acc_gnb = metrics.accuracy_score(y_test, y_pred_gnb)
prec_gnb = metrics.precision_score(y_test, y_pred_gnb)
rec_gnb = metrics.recall_score(y_test, y_pred_gnb)

# ss_train_test = StandardScaler()
# X_train_ss_scaled = ss_train_test.fit_transform(X_train)
# X_test_ss_scaled = ss_train_test.transform(X_test)

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

# l_regressor_ss = LinearRegression()
# l_regressor_ss.fit(X_train_ss_scaled, y_train)
# y_pred_l_reg_ss = l_regressor_ss.predict(X_test_ss_scaled)


# st.write("""
#     ## Dengan Menggunakan Random Forest Diperoleh Skor Untuk Data Test

# """)

st.write("Maka Dengan Menggunakan Naive Bayes Diperoleh Skor Accuracy")
st.write(acc_gnb)
st.write("Maka Dengan Menggunakan Naive Bayes Diperoleh Skor Precision")
st.write(prec_gnb)
st.write("Maka Dengan Menggunakan Naive Bayes Diperoleh Skor Recall")
st.write(rec_gnb)


st.write("# Sekarang Silahkan Masukan Data Untuk Mengetahui Prediksi Peluang Apakah Kamu Positif Atau Negatif Diabetes")


inputPregnancies = st.number_input("Masukan Pregnancies Score: ")
inputGlucose = st.number_input("Masukan Glucose Score: ")
inputBP = st.number_input("Masukan Blood Pressure: ")
inputST = st.number_input("Masukan Skin Thickness: ")
inputInsulin = st.number_input("Masukan Insulin: ")
inputBMI = st.number_input("Masukan BMI: ")
inputBPF = st.number_input("Masukkan Diabetes Pedigree Function: ")
inputAge = st.number_input("Umur :")
	
with st.expander("Prediction Results"):
    completeData = np.array([inputPregnancies, inputGlucose, inputBP, 
                            inputST, inputInsulin, inputBMI, inputBPF, inputAge]).reshape(1, -1)
    scaledData = scaler.transform(completeData)
    prediction = gnb.predict(scaledData)
    st.write('Tekan Submit Untuk Melihat Prediksi Peluang Terkena Diabetes Anda')
    if prediction == 1 :
        st.write(prediction, "## Anda Positif Diabetes") 
    else:
        st.write(prediction, "## Anda Negatif Diabetes")
    
# prediction = gnb.predict(scaledData)
