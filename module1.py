import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split

# 1. Завантаження даних
print("Loading train.csv...")
train = pd.read_csv('train.csv')
print(f"Data loaded! Total rows: {len(train)}")

# 2. Виділяємо мітки (labels) та пікселі (features)
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1) 

# --- ПІДГОТОВКА ДАНИХ (ЗАВДАННЯ №2) ---

# Нормалізація: переводимо пікселі з 0-255 у діапазон 0-1 для легшого навчання
X_train = X_train / 255.0

# Reshape: перетворюємо пласкі списки у матриці 28x28x1 (формат для нейромереж CNN)
X_train = X_train.values.reshape(-1, 28, 28, 1)

# Розподіл: виділяємо 10% даних для перевірки (валідації)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

print("Preprocessing complete.")
print(f"Train set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# --- ПЕРЕВІРКА РЕЗУЛЬТАТУ ---

# Обираємо випадковий індекс зі списку тренувальних даних
random_index = random.randint(0, len(X_train) - 1)

# Малюємо випадкову цифру
plt.imshow(X_train[random_index][:,:,0], cmap='gray')
plt.title(f"Label (Random Index {random_index}): {Y_train.iloc[random_index]}")

print(f"Showing random image at index {random_index}...")
plt.show()