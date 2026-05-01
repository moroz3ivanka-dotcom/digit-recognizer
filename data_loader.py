import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    
    X = train.drop(labels=["label"], axis=1)
    y = train["label"]

    # Балансування класів (як у прикладі з Kaggle)
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Нормалізація (0-1)
    X_resampled = X_resampled / 255.0

    # Розподіл на Train/Test
    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)