from pydataset import data
import seaborn as sns
import pandas as pd
import acquire

def prep_iris():
    df = acquire.get_iris_data()
    df = df.drop(columns=['species_id', 'measurement_id'], axis = 1)
    dummy_df = pd.get_dummies(df['species_name'], dummy_na=False)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_titanic():

    df = acquire.get_titanic_data()
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age'])
    df['embark_town'] = df.embark_town.dropna()
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_telco():

    df = acquire.get_telco_data()
    df = df.drop_duplicates()
    df = df.drop(columns=['Unnamed: 0', 'payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'])
    #dummy_df = pd.get_dummies(df[[]], drop_first=True)
    #df = pd.concat([df, dummy_df], axis=1)
    return df