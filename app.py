import tensorflow
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
import pandas as pd

model=load_model('model.weights.h5')

#with open('le.pkl', "rb") as file:
    #le = pickle.load(file)

#with open('ohe.pkl','rb') as file:
    #ohe=pickle.load(file)

#with open('scaler.pkl','rb') as file:
   #scaler=pickle.load(file)

le=pickle.load(open('le.sav','rb'))
ohe=pickle.load(open('ohe.sav','rb'))
scaler=pickle.load(open('scaler.sav','rb'))

st.title('Churn Prediction')
CreditScore=st.number_input('Credit Score')
Geography=st.selectbox("Geography",ohe.categories_[0])
Gender=st.selectbox('Gender',le.classes_)
Age=st.slider('Age',18,100)
Tenure=st.slider("Tenure",1,10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('No of Products:',1,4)
HasCrCard=st.selectbox('Has Credit Card',[0,1])
IsActiveMember=st.selectbox('Is Active Member',[0,1])
EstimatedSalary=st.number_input('Estimated Salary')

input_data={
    'CreditScore': [CreditScore],
    'Gender': [le.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
}

df=pd.DataFrame(input_data)
encoded=ohe.transform([[Geography]]).toarray()
dff=pd.DataFrame(encoded,columns=ohe.get_feature_names_out(['Geography']))
df=pd.concat([df,dff],axis=1)
scaled_df=scaler.transform(df)

prediction=model.predict(scaled_df)
probability=prediction[0][0]

st.write('Probability',probability)
if probability>0.5:
    st.write('The Customer is likely to churn')
else:
    st.write('The Customer is not likely to churn')