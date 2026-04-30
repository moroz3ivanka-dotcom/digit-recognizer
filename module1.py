import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Завантаження даних
print("Loading train.csv...")
train = pd.read_csv('train.csv')
print(f"Data loaded! Number of rows: {len(train)}")

# 2. Виділяємо мітки (labels) та пікселі (features)
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1) 

# --- ПІДГОТОВКА (ЗАВДАННЯ №2) ---

# Нормалізація: переводимо значення пікселів з діапазону 0-255 у 0-1
X_train = X_train / 255.0

# Reshape: перетворюємо дані у формат матриць 28x28 з 1 кольоровим каналом
X_train = X_train.values.reshape(-1, 28, 28, 1)

# Розподіл даних: 90% для навчання, 10% для перевірки (валідації)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

print(f"Preprocessing complete. Array shape: {X_train.shape}")
print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

# --- ПЕРЕВІРКА ---
# Візуалізація цифри під індексом 5 для контролю результату
plt.imshow(X_train[5][:,:,0], cmap='gray')
plt.title(f"Label (Index 5): {Y_train.iloc[5]}")
plt.show()