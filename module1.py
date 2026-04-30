import pandas as pd
import matplotlib.pyplot as plt

# 1. Читаємо файл прямо з папки
print("Зчитую файл train.csv...")
train = pd.read_csv('train.csv')

# 2. Виводимо інформацію
print(f"Дані завантажено! Кількість рядків: {len(train)}")

# 3. Беремо перший рядок (це картинка однієї цифри)
# Перша колонка - це сама цифра (label), решта 784 - це пікселі
label = train.iloc[0, 0]
pixels = train.iloc[0, 1:].values.reshape(28, 28)

# 4. Малюємо картинку
plt.imshow(pixels, cmap='gray')
plt.title(f"Це цифра: {label}")
plt.show()