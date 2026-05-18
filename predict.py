import numpy as np
from keras.models import load_model

def predict_single_image(model_path, image_pixels):

    # 1. Завантажуємо навчену модель
    model = load_model(model_path)
    
    # Перетворюємо в numpy масив для зручної роботи
    img_array = np.array(image_pixels, dtype="float32")
    
    # 2. Нормалізація (якщо дані прийшли у форматі 0-255, ділимо на 255.0)
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
        
    # 3. Приводимо до розміру (1, 28, 28, 1) - формат, який вимагає ваша CNN нейромережа
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # 4. Прогноз моделі (повертає ймовірності для кожної з 10 цифр)
    prediction = model.predict(img_array)
    
    # 5. Той самий argmax, про який казав викладач (вибираємо індекс з найбільшою ймовірністю)
    result = np.argmax(prediction, axis=1)[0]
    
    return int(result)

# Блок перевірки: запуститься тільки якщо ти запускаєш саме цей файл
if __name__ == "__main__":
    # Створюємо тестову "пусту" картинку (784 нулі), щоб перевірити чи немає помилок у коді
    test_fake_image = np.zeros(784)
    try:
        digit = predict_single_image('my_mnist_model.h5', test_fake_image)
        print(f"Script executed successfully! The model recognized the test as digit: {digit}")
    except Exception as e:
        print(f"Error during script verification: {e}")