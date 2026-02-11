import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

# Charger le modèle entraîné
model = tf.keras.models.load_model("models/mnist_cnn.h5")

# Fonction pour préparer l'image dessinée
def preprocess_image(pil_img):
    arr = np.array(pil_img.resize((28,28)).convert('L'))
    arr = 1.0 - arr/255.0  # inversion si fond blanc
    arr = arr.reshape(1,28,28,1)
    return arr

# Fenêtre Tkinter pour dessiner
class DrawCanvas(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Dessiner un chiffre")
        self.canvas_width = 280
        self.canvas_height = 280
        self.brush_radius = 8
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        tk.Button(self, text="Prédire", command=self.predict).pack(side="left", padx=10)
        tk.Button(self, text="Effacer", command=self.clear).pack(side="left", padx=10)
        tk.Button(self, text="Fermer", command=self.destroy).pack(side="left", padx=10)

    def paint(self, event):
        x1, y1 = (event.x - self.brush_radius), (event.y - self.brush_radius)
        x2, y2 = (event.x + self.brush_radius), (event.y + self.brush_radius)
        self.canvas.create_oval(x1,y1,x2,y2, fill="black")
        self.draw.ellipse([x1,y1,x2,y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,self.canvas_width,self.canvas_height], fill=255)

    def predict(self):
        arr = preprocess_image(self.image)
        digit = np.argmax(model.predict(arr))
        messagebox.showinfo("Résultat", f"Chiffre prédit : {digit}")

# Fenêtre principale
root = tk.Tk()
root.title("MNIST - Dessiner un chiffre")
tk.Button(root, text="Dessiner un chiffre", command=lambda: DrawCanvas(root)).pack(pady=20)
tk.Button(root, text="Quitter", command=root.quit).pack(pady=10)
root.mainloop()
