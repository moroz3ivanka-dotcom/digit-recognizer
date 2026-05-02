import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# 1. Завантаження вже навченої моделі
print("Loading saved model...")
model = load_model('my_mnist_model.h5')

# 2. Завантаження тестових даних для Kaggle
print("Loading test.csv...")
test_data = pd.read_csv('test.csv')

# 3. Передобробка (точно так само, як ми вчили модель)
print("Preprocessing data...")
test_rescaled = test_data / 255.0
test_rescaled = test_rescaled.values.reshape(-1, 28, 28, 1)

# 4. Прогноз
print("Predicting digits for Kaggle...")
predictions = model.predict(test_rescaled)
results = np.argmax(predictions, axis=1)

# 5. Створення фінального файлу
submission = pd.DataFrame({
    "ImageId": range(1, len(results) + 1),
    "Label": results
})

submission.to_csv("submission.csv", index=False)
print("\n--- DONE! ---")
print("File 'submission.csv' has been created and is ready for upload to Kaggle.")