import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import to_categorical
from tkinter import Tk, Canvas, Button, Label, mainloop
from PIL import ImageGrab, Image
import numpy as np
import os

class HandwrittenDigitRecognizer:
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        # Modeli yükle ya da yeniden eğit
        if os.path.exists("mnist_model.h5"):
            self.model = load_model("mnist_model.h5")
            print("Model başarıyla yüklendi!")
        else:
            print("Model bulunamadı, yeniden eğitiliyor...")
            self._train_model()

    def _train_model(self):
        # MNIST veri setini yükle ve işle
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # Model yapısı
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        # Modeli eğit
        self.model.fit(x_train, y_train, batch_size=128, epochs=5, 
                       validation_data=(x_test, y_test), verbose=1)
        self.model.save("mnist_model.h5")
        print("Model başarıyla eğitildi ve kaydedildi!")

    def predict_digit(self, img):
        # Görüntüyü işle
        img = img.resize((28, 28)).convert('L')
        img = np.array(img).reshape(1, 28, 28, 1) / 255.0
        prediction = self.model.predict(img)[0]
        return np.argmax(prediction), max(prediction)

    def start_gui(self):
        app = DigitRecognizerApp(self)
        mainloop()

class DigitRecognizerApp(Tk):
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.title("El Yazısı Rakam Tanıyıcı")
        self.geometry("400x400")
        self.canvas = Canvas(self, width=300, height=300, bg='white', cursor="cross")
        self.label = Label(self, text="Bir rakam çizin!", font=("Helvetica", 18))
        self.predict_button = Button(self, text="Tanı", command=self.classify_digit, font=("Helvetica", 12))
        self.clear_button = Button(self, text="Temizle", command=self.clear_canvas, font=("Helvetica", 12))

        # Yerleşim
        self.canvas.grid(row=0, column=0, columnspan=2, pady=10)
        self.label.grid(row=1, column=0, columnspan=2, pady=10)
        self.predict_button.grid(row=2, column=1, pady=10)
        self.clear_button.grid(row=2, column=0, pady=10)
        self.canvas.bind("<B1-Motion>", self._draw)

    def clear_canvas(self):
        self.canvas.delete("all")

    def classify_digit(self):
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x1, y1))
        digit, acc = self.recognizer.predict_digit(img)
        self.label.config(text=f"Tahmin: {digit} ({int(acc * 100)}%)")

    def _draw(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')

if __name__ == "__main__":
    recognizer = HandwrittenDigitRecognizer()
    recognizer.start_gui()
