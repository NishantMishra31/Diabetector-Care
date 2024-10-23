import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("./diabetes.csv")
print(data.head(10))

print(f"Shape of the data: {data.shape}")

print(f"Are there null values? {data.isnull().values.any()}")

data.rename(columns={'DiabetesPedigreeFunction':'DPF', 'BloodPressure':'BP'}, inplace=True)
print(data.head(5))

print(data.describe())

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10, 10))
sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()


for column in ['Glucose', 'BP', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']:
    print(f"Number of zeros in {column} : {data[data[column]==0].shape[0]}")


data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())
data['BP'] = data['BP'].replace(0, data['BP'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].mean())
data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())
data['DPF'] = data['DPF'].replace(0, data['DPF'].mean())
data['Age'] = data['Age'].replace(0, data['Age'].mean())


for column in ['Glucose', 'BP', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']:
    print(f"Number of zeros in {column} after replacement: {data[data[column]==0].shape[0]}")


positive_outcome = len(data.loc[data["Outcome"]==1])
negative_outcome = len(data.loc[data["Outcome"]==0])
print(f"Diabetic: {positive_outcome}, Non-diabetic: {negative_outcome}")


y = np.array([positive_outcome, negative_outcome])
mylabels = ["Diabetic people (268)", "Non-diabetic people (500)"]
plt.pie(y, labels=mylabels, colors=["orange", "yellow"])
plt.title("Number of diabetic and Non-diabetic persons")
plt.show()


df = {'Diabetic': positive_outcome, 'Non-diabetic': negative_outcome}
A = list(df.keys())
B = list(df.values())
plt.bar(A, B, width=0.2)
plt.title("Number of diabetic and Non-diabetic persons")
plt.show()


X = data.drop(columns=["Outcome"])
Y = data["Outcome"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=10)


model = RandomForestClassifier(random_state=10)
model.fit(X_train, Y_train.ravel())


pred = model.predict(X_test)


acc = metrics.accuracy_score(Y_test, pred)
print(f"\n\n ACCURACY OF THE MODEL : {acc:.2f}\n")


def prediction_calculator(n):
    for i in range(n):
        print(f"\nENTER THE DETAILS FOR PERSON : {i+1}")
        Gender_ip = input("\nGENDER M/F/m/f: ").strip().lower()
        Preg_ip = 0 if Gender_ip == 'm' else int(input("Number of Pregnancies : "))
        Age_ip = float(input("Age : "))
        Bmi_ip = float(input("BMI : "))
        Glucose_ip = float(input("Glucose level : "))
        Insulin_ip = float(input("Insulin level : "))
        Bp_ip = float(input("BP level : "))
        St_ip = float(input("Skin Thickness : "))
        Dpf_ip = float(input("Diabetes Pedigree Function : "))
        
        c = np.array([Preg_ip, Glucose_ip, Bp_ip, St_ip, Insulin_ip, Bmi_ip, Dpf_ip, Age_ip])
        c_rs = c.reshape(1, -1)
        pred = model.predict(c_rs)
        
        if pred == 1:
            print("DIABETIC PERSON !!")
        else:
            print("NON-DIABETIC PERSON :)")

no_of_people = int(input("\n ENTER NUMBER OF PEOPLE : "))
prediction_calculator(no_of_people)