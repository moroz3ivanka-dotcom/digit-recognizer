import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_and_preprocess_data

def run_full_evaluation(model_path, num_samples=100):
    # 1. Load Data and Model
    print("Loading data and trained model...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = load_model(model_path)

    # 2. Select a batch of images for implementation demo 
    # Taking the first 100 images for batch processing
    X_sample = X_test[:num_samples]
    y_true = y_test[:num_samples]

    # Reshape for CNN input (Batch, Height, Width, Channels)
    X_sample_cnn = X_sample.values.reshape(-1, 28, 28, 1)

    # 3. Batch Prediction Implementation
    print(f"Executing batch prediction for {num_samples} images...")
    predictions = model.predict(X_sample_cnn)
    y_pred = np.argmax(predictions, axis=1)

    # 4. Evaluation Metrics 
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT:")
    print("="*60)
    # This report shows Precision, Recall, and F1-score for each digit (0-9)
    print(classification_report(y_true, y_pred))

    # 5. Visualizing the Implementation
    print("Generating visualization of predictions...")
    plt.figure(figsize=(15, 10))
    for i in range(15):  # Showing first 15 results as a sample
        plt.subplot(3, 5, i + 1)
        img = X_sample.iloc[i].values.reshape(28, 28)
        plt.imshow(img, cmap='gray')
        
        color = 'green' if y_pred[i] == y_true.iloc[i] else 'red'
        plt.title(f"Actual: {y_true.iloc[i]}\nPredicted: {y_pred[i]}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # 6. Confusion Matrix Research
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix - Research on {num_samples} Samples')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    MODEL_FILE = 'my_mnist_model.h5'
    run_full_evaluation(MODEL_FILE)