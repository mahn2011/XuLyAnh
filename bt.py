import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk
from tkinter import filedialog

# Tải bộ dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Tăng cường dữ liệu
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# Xây dựng và huấn luyện mô hình
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(x_train, to_categorical(y_train, 10), batch_size=128),
          epochs=15, validation_data=(x_test, to_categorical(y_test, 10)))
model.save('mnist_model.keras')  # Lưu mô hình

# Hàm tiền xử lý ảnh
def preprocess_image(img):
    img = img.convert('L')  # Chuyển sang ảnh mức xám
    img = img.resize((28, 28), Image.LANCZOS)  # Đổi kích thước
    img = ImageOps.invert(img)  # Đảo màu
    img = ImageOps.autocontrast(img)  # Tăng độ tương phản tự động
    img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255
    return img_array

# Hàm xử lý ảnh từ file và dự đoán
def predict_image_from_file(filepath):
    img = Image.open(filepath)
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]
    print(f'Dự đoán từ file ảnh: {predicted_label}, Độ tin cậy: {confidence:.2f}')
    plt.imshow(img_array[0].reshape(28, 28), cmap='gray')
    plt.title(f'Dự đoán: {predicted_label}, Độ tin cậy: {confidence:.2f}')
    plt.axis('off')
    plt.show()

# Tạo giao diện để vẽ số
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()
        
        self.image = Image.new('L', (200, 200), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind('<B1-Motion>', self.paint)
        tk.Button(root, text='Dự đoán', command=self.predict_drawn_digit).pack()
        tk.Button(root, text='Chọn ảnh', command=self.load_image).pack()  # Nút chọn ảnh
        tk.Button(root, text='Xóa', command=self.clear_canvas).pack()

    def paint(self, event):
        x, y = event.x, event.y
        r = 6  # bán kính của nét vẽ
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (200, 200), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def predict_drawn_digit(self):
        img = self.image.resize((28, 28), Image.LANCZOS)
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        confidence = prediction[0][predicted_label]
        print(f'Dự đoán từ hình vẽ: {predicted_label}, Độ tin cậy: {confidence:.2f}')
        plt.imshow(img_array[0].reshape(28, 28), cmap='gray')
        plt.title(f'Dự đoán: {predicted_label}, Độ tin cậy: {confidence:.2f}')
        plt.axis('off')
        plt.show()

    def load_image(self):
        # Mở hộp thoại chọn file ảnh
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if filepath:  # Kiểm tra xem người dùng đã chọn file hay chưa
            predict_image_from_file(filepath)

# Khởi chạy ứng dụng
root = tk.Tk()
app = DrawApp(root)
root.mainloop()
