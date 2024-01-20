# Lab21 : Prédiction de crédit logement
# Réalisé par : Hayat el allaouy /Emsi 2023-2024
# Référence : https://www.kaggle.com/code/rodsonzepekinio/pr-vision-d-un-cr-dit-logement

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Step 1 : DataSet
# Data manipulation pandas
dt = pd.read_csv("datasets/train.csv")# load dataset
print(dt.head())
print(dt.info())
print(dt.isna().sum())
"""
Remplacer les variables manquantes categoriques par leurs modes
Remplacer les variables manquantes numériques par la médiane
"""
def trans(data): # Data transformation
    for c in data.columns:
        if data[c].dtype=='int64' or data[c].dtype=='float64':
            data[c].fillna(data[c].median(),inplace=True)
        else:
              data[c].fillna(data[c].mode()[0],inplace=True)
trans(dt)
print(dt.isna().sum())
# Data transformation
var_cat = dt[["Loan_Status","Gender","Married","Dependents","Education","Self_Employed","Property_Area","Credit_History"]]
var_num = dt[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]]
original_dataset = pd.concat([var_cat,var_num],axis=1)
print(original_dataset.head())
var_cat=pd.get_dummies(var_cat,drop_first=True)
transformed_dataset = pd.concat([var_cat,var_num],axis=1)
print(transformed_dataset.head())
print(transformed_dataset.info())
print(transformed_dataset.isna().sum())

# Data visualization
print(dt["Loan_Status"].value_counts(normalize=True)*100)
fig = px.histogram(dt, x="Loan_Status",title='Crédit accordé ou pas', color="Loan_Status",template= 'plotly_dark')
# fig.show(font = dict(size=17,family="Franklin Gothic"))
fig = px.pie(dt, names="Dependents",title='Dependents',color="Dependents",template= 'plotly_dark')
# fig.show(font = dict(size=17,family="Franklin Gothic"))


# Save transformed dataset
transformed_dataset.to_csv('datasets/transformed_dataset.csv')

# Split transformed_dataset on Features X, Target y
y = transformed_dataset["Loan_Status_Y"]
X = transformed_dataset.drop("Loan_Status_Y",axis=1)

# Split transformed_dataset on train 80% test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Step 2 : Model
model = LogisticRegression()

# Step 3 : Train
model.fit(X_train,y_train)

# Step 4 : Test
print("Votre Intelligence Arti est fiable à", model.score(X_test,y_test)*100,"%")