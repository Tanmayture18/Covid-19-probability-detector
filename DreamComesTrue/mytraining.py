#Machine learning model

# Importing necessary libraries
import pandas as pd
import numpy as np
import pickle





if __name__=='__main__':
    df=pd.read_csv('tdtt.csv')
    # Dropping unnecessary columns
    df.drop(['batch_date', 'test_name', 'swab_type','high_risk_exposure_occupation', 'high_risk_interactions','rapid_flu_results',
       'rapid_strep_results', 'ctab', 'labored_respiration', 'rhonchi',
       'wheezes', 'days_since_symptom_onset','cxr_findings', 'cxr_impression', 'cxr_label',
       'cxr_link', 'er_referral','smoker'],axis=1,inplace=True)

    # Null values imputation
    df['temperature']=df['temperature'].fillna(df['temperature'].mean())
    df['pulse']=df['pulse'].fillna(df['pulse'].mean())
    df['sys']=df['sys'].fillna(df['sys'].mean())
    df['dia']=df['dia'].fillna(df['dia'].mean())
    df['rr']=df['rr'].fillna(df['rr'].mean())
    df['sats']=df['sats'].fillna(df['sats'].mean())
    df['cough_severity']=df['cough_severity'].fillna(df['cough_severity'].mode()[0])
    df['fever']=df['fever'].fillna(df['fever'].mode()[0])
    df['sob']=df['sob'].fillna(df['sob'].mode()[0])
    df['sob_severity']=df['sob_severity'].fillna(df['sob_severity'].mode()[0])
    df['diarrhea']=df['diarrhea'].fillna(df['diarrhea'].mode()[0])
    df['fatigue']=df['fatigue'].fillna(df['fatigue'].mode()[0])
    df['headache']=df['headache'].fillna(df['headache'].mode()[0])
    df['loss_of_smell']=df['loss_of_smell'].fillna(df['loss_of_smell'].mode()[0])
    df['loss_of_taste']=df['loss_of_taste'].fillna(df['loss_of_taste'].mode()[0])
    df['runny_nose']=df['runny_nose'].fillna(df['runny_nose'].mode()[0])
    df['muscle_sore']=df['muscle_sore'].fillna(df['muscle_sore'].mode()[0])
    df['sore_throat']=df['sore_throat'].fillna(df['sore_throat'].mode()[0])

# Label encoding for categorical variables
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    df['covid19_test_results']=le.fit_transform(df['covid19_test_results'])
    df['diabetes']=le.fit_transform(df['diabetes'])
    df['chd']=le.fit_transform(df['chd'])
    df['htn']=le.fit_transform(df['htn'])
    df['cancer']=le.fit_transform(df['cancer'])
    df['asthma']=le.fit_transform(df['asthma'])
    df['copd']=le.fit_transform(df['copd'])
    df['autoimmune_dis']=le.fit_transform(df['autoimmune_dis'])
    df['fever']=le.fit_transform(df['fever'])
    df['cough_severity']=le.fit_transform(df['cough_severity'])
    df['cough']=le.fit_transform(df['cough'])
    df['sob']=le.fit_transform(df['sob'])
    df['sob_severity']=le.fit_transform(df['sob_severity'])
    df['fatigue']=le.fit_transform(df['fatigue'])
    df['headache']=le.fit_transform(df['headache'])
    df['diarrhea']=le.fit_transform(df['diarrhea'])
    df['loss_of_smell']=le.fit_transform(df['loss_of_smell'])
    df['loss_of_taste']=le.fit_transform(df['loss_of_taste'])
    df['runny_nose']=le.fit_transform(df['runny_nose'])
    df['muscle_sore']=le.fit_transform(df['muscle_sore'])
    df['sore_throat']=le.fit_transform(df['sore_throat'])

# Dropping unnecessary columns
    df.drop(['htn','asthma','copd','autoimmune_dis','cough_severity','sob','sob_severity','fatigue'],axis=1,inplace=True)

# Separating dependant and independant variables
    X=df.drop('covid19_test_results',axis=1)
    y=df.covid19_test_results

# performing train-test split
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
  

# Fitting logistic regression algorithm
    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression()
    clf.fit(X_train,y_train)

#dump information form file
    file=open('model.pk1','wb')
    pickle.dump(clf,file)   
    file.close() 


