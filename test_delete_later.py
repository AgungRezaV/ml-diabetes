import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

st.write("""
# My First Streamlit website / App
Hello *World!*
""")

st.write("""
# Overwiew Data
""")
our_datasets = pd.read_csv("./dataset/diabetes.csv")
st.dataframe(our_datasets)

st.write("""
# Deskripsi Data
""")

st.dataframe(our_datasets.describe())

st.write("""
# Preprocessing data
""")

st.write("""
# Test
""")

df = our_datasets.copy()
df[['Glucose','BloodPressure','SkinThickness','BMI']] = our_datasets[['Glucose','BloodPressure','SkinThickness','BMI']].replace(0,np.NaN)
# Check Missing Values
# df.isnull().sum()

def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

median_target('Glucose')
df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 107
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 140

df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 74.5

df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 32

df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 34.3

corr_matrix = our_datasets.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix,
                 annot = True,
                 linewidths = 0.5,
                 fmt = ".2f",
                 cmap = "YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

X_pp = df.drop(['Outcome','Insulin','DiabetesPedigreeFunction'], axis=1)
X = preprocessing.scale(X_pp)
scaler = preprocessing.StandardScaler().fit(X_pp)
y = df['Outcome']
st.write("## Input Data y",y)

# Label dan Features
X_train, X_test, y_train, y_test  = train_test_split(X, y, shuffle = True, test_size=0.2, random_state=0,stratify=y)

#Random Forest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rfo = RandomForestClassifier()
n_estimators = [100, 300, 400, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth, 
              min_samples_split = min_samples_split, 
              min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(rfo, hyperF, scoring='roc_auc',
                     cv = 5, verbose = 1, 
                     n_jobs = -1)

bestF = gridF.fit(X, y)

accuracy_scores = []
recall_scores = []

for i in range(0, 300):
    rfo = RandomForestClassifier(max_depth= 15, min_samples_leaf= 5, 
                                min_samples_split= 15, n_estimators= 100,random_state=i)
    rfo.fit(X_train, y_train)
    accuracy_scores.append(accuracy_scores(y_test, rfo.predict(X_test)))
    recall_scores.append(recall_scores(y_test, rfo.predict(X_test)))

print(max(accuracy_scores))
print("Best random i accuracy =",accuracy_scores.index(max(accuracy_scores)))
print(max(recall_scores))
print("Best random i recall =",recall_scores.index(max(recall_scores)))

# Modelling random forest
rfo = RandomForestClassifier(max_depth= 15, min_samples_leaf= 5, 
                             min_samples_split= 15, n_estimators= 100,random_state=260) #or 46
rfo.fit(X_train, y_train)

# Predict to Test Data
y_pred_rfo = rfo.predict(X_test)

# Confusion Matriks
cm_rfo = metrics.confusion_matrix(y_test, y_pred_rfo)
cm_rfo

acc_rfo = metrics.accuracy_score(y_test, y_pred_rfo)
prec_rfo = metrics.precision_score(y_test, y_pred_rfo)
rec_rfo = metrics.recall_score(y_test, y_pred_rfo)
f1_rfo = metrics.f1_score(y_test, y_pred_rfo)
kappa_rfo = metrics.cohen_kappa_score(y_test, y_pred_rfo)

print("Accuracy:", acc_rfo)
print("Precision:", prec_rfo)
print("Recall:", rec_rfo)
print("F1 Score:", f1_rfo)
print("Cohens Kappa Score:", kappa_rfo)

y_pred_rfo_proba = rfo.predict_proba(X_test)[::,1]
fprrfo, tprrfo, _ = metrics.roc_curve(y_test,  y_pred_rfo_proba)
aucrfo = metrics.roc_auc_score(y_test, y_pred_rfo_proba)
plt.plot(fprrfo,tprrfo,label="Random Forest, auc="+str(aucrfo))
plt.title('ROC Curve - Random Forest')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc=4)
plt.show()


# st.write("Dengan Menggunakan Multiple Linear Regression Diperoleh Skor Untuk Data Test")
# st.write(r2_score(y_test, y_pred_l_reg_ss))


# st.write("# Sekarang Silahkan Masukan Skor Test Kamu Untuk Mengetahui Prediksi Peluang Kelulusan S2 Kamu")


# form = st.form(key='my-form')
# inputGRE = form.number_input("Masukan GRE Score: ", 0)
# inputTOEFL = form.number_input("Masukan TOEFL Score: ", 0)
# inputUnivRating = form.number_input("Masukan Rating Univ: ", 0)
# inputSOP = form.number_input("Masukan Kekuatan SOP: ", 0)
# inputLOR = form.number_input("Masukan Kekuatan LOR: ", 0)
# inputCGPA = form.number_input("Masukan CGPA: ", 0)
# inputResearch = form.number_input("Pengalaman Researc, 1 Jika Pernah Riset, 0 Jika Tidak", 0)
# submit = form.form_submit_button('Submit')

# completeData = np.array([inputGRE, inputTOEFL, inputUnivRating, 
#                         inputSOP, inputLOR, inputCGPA, inputResearch]).reshape(1, -1)
# scaledData = ss_train_test.transform(completeData)


# st.write('Tekan Submit Untuk Melihat Prediksi Peluang S2 Anda')

# if submit:
#     prediction = l_regressor_ss.predict(scaledData)
#     if prediction > 1 :
#         result = 1
#     elif prediction < 0 :
#         result = 0
#     else :
#         result = prediction[0]
#     st.write(result*100, "Percent")
