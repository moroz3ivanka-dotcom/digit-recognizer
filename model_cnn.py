import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from data_loader import load_and_preprocess_data

# 1. Завантаження даних
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Reshape для нейронки
X_train_cnn = X_train.values.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.values.reshape(-1, 28, 28, 1)

# 2. Будуємо архітектуру CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 3. Навчання (зберігаємо історію в змінну history для графіків)
print("Training CNN...")
history = model.fit(X_train_cnn, y_train, epochs=3, 
                    validation_split=0.1, batch_size=64)
model.save('my_mnist_model.h5')
print("Model saved to my_mnist_model.h5")

# 4. Оцінка моделі
loss, acc = model.evaluate(X_test_cnn, y_test)
print(f"\nFinal CNN Accuracy on Test Set: {acc * 100:.2f}%")

# 5. ВІЗУАЛІЗАЦІЯ 1: Графік точності
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 6. ВІЗУАЛІЗАЦІЯ 2: Матриця помилок
plt.subplot(1, 2, 2)
predictions = model.predict(X_test_cnn)
y_pred_classes = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - CNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()