import numpy as np 
import pandas as pd
import joblib 
from sklearn.preprocessing import StandardScaler

def load_data(filepath) : 
    return pd.read_csv(filepath)

def remove_outlier(df) : 
    columns = ['age', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'oldpeak','slope','ca']

    for column in columns : 
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1 

        lower_bound = Q1 - 1.5 * IQR 
        upper_bound = Q3 + 1.5 * IQR 

        median = df[column].median()
        df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)


def preprocess(df): 
    X = df.drop('condition', axis=1)
    y = df['condition']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler,'preprocessing/scaler.pkl')

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    df_final = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

    return df_final

def save_processed(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    df_raw = load_data('preprocessing/HeartDiagnosa_raw.csv')  
    df_processed = preprocess(df_raw)
    save_processed(df_processed, 'preprocessing/HeartDiagnosa_preprocessing.csv')
