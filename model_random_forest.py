from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data

# 1. Завантажуємо дані через наш спільний завантажувач
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# 2. Навчаємо модель (Random Forest)
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 3. Робимо прогноз на тестових даних
print("Evaluating model...")
predictions = rf_model.predict(X_test)

# 4. Виводимо текстовий звіт
accuracy = accuracy_score(y_test, predictions)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 5. Візуалізація: Матриця помилок
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Random Forest')
plt.show()