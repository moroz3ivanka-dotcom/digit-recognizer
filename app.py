import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from predict import predict_single_image

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

st.title("🖌️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) clearly in the center!")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=20, 
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def process_digit(img):
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]
        
        # Створюємо квадратний фон
        size = max(w, h) + 40
        centered_img = np.zeros((size, size), dtype="uint8")
        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        centered_img[offset_y:offset_y+h, offset_x:offset_x+w] = digit
        
        # Стискаємо до 20x20
        final_img = cv2.resize(centered_img, (20, 20), interpolation=cv2.INTER_AREA)
        
        # РОБИМО ЛІНІЮ ТОВСТОЮ (Dilation)
        kernel = np.ones((2,2), np.uint8)
        final_img = cv2.dilate(final_img, kernel, iterations=1)
        
        # Додаємо рамку до 28x28
        final_img = cv2.copyMakeBorder(final_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        return final_img
    return cv2.resize(img, (28, 28))

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        img_final = process_digit(img)
        pixels = img_final.reshape(784)
        
        try:
            result = predict_single_image('my_mnist_model.h5', pixels)
            st.success(f"### Result: The model predicted the digit: **{result}**")
            st.image(img_final, width=100, caption="What the model sees")
        except Exception as e:
            st.error(f"Error: {e}")