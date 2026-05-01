import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from data_loader import load_and_preprocess_data

# 1. Get data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

print("Starting K-Means Clustering (this may take a minute)...")
# 2. Train KMeans
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train)

# 3. "Magic" step: Map each cluster to the most frequent real label
# Оскільки KMeans не знає назв цифр, ми самі їх призначаємо
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    if np.any(mask):
        # Знаходимо, яка цифра найчастіше зустрічається в цьому кластері
        labels[mask] = mode(y_train[mask], keepdims=True)[0]

# 4. Evaluation on Test Set
test_clusters = kmeans.predict(X_test)
test_labels = np.zeros_like(test_clusters)
for i in range(10):
    mask = (test_clusters == i)
    if np.any(mask):
        # Використовуємо ті ж самі назви кластерів для тесту
        cluster_mask = (clusters == i)
        if np.any(cluster_mask):
            test_labels[mask] = mode(y_train[cluster_mask], keepdims=True)[0]

accuracy = accuracy_score(y_test, test_labels)
print(f"K-Means Clustering Accuracy: {accuracy * 100:.2f}%")