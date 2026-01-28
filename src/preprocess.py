import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal', 'overweight', 'obese'])
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 40, 55, 100], labels=['young', 'adult', 'middle', 'senior'])
    df['is_smoker'] = (df['smoker'] == 'yes').astype(int)
    df['high_risk'] = ((df['smoker'] == 'yes') & (df['bmi'] > 30)).astype(int)
    
    df = df.rename(columns={'smoker': 'is_smoker'})

    df = pd.get_dummies(df, columns=['sex', 'region', 'bmi_category', 'age_group'], drop_first=True)
       
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)