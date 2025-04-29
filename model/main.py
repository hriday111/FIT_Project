import os
import json
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
# -----------------------------
# Step 1: Load your JSON mapping
# -----------------------------
# e.g. labels.json contains:
# {"cont17.png":"u","cont18.png":"B", ... }
json_path = 'progress.json'
with open(json_path, 'r') as f:
    labels_dict = json.load(f)

# -----------------------------
# Step 2: Load and preprocess images from output/
# -----------------------------
images_folder = 'output'     # your folder name
target_size = (28, 28)       # adjust if needed

X, Y = [], []
for fname, label in labels_dict.items():
    img_path = os.path.join(images_folder, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: could not load {img_path}")
        continue
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, -1)  # shape: (28,28,1)
    X.append(img)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

# -----------------------------
# Step 3: Encode labels and split
# -----------------------------
encoder = LabelEncoder()
Y_int = encoder.fit_transform(Y)              # e.g. 'A'→0, 'B'→1, …
Y_onehot = tf.keras.utils.to_categorical(Y_int)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_onehot, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Build the CNN
# -----------------------------
num_classes = Y_onehot.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# -----------------------------
# Step 5: Train & Evaluate
# -----------------------------
history = model.fit(
    X_train, Y_train,
    epochs=10,
    validation_data=(X_test, Y_test)
)

loss, acc = model.evaluate(X_test, Y_test)
print(f"Test accuracy: {acc:.4f}")

# -----------------------------
# Step 6: Prediction function
# -----------------------------
def predict_character(img=None, image_path=None):
    #if(img is None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)  # batch dim
    preds = model.predict(img)
    idx = np.argmax(preds, axis=1)[0]
    return encoder.inverse_transform([idx])[0]

# Example:
print(predict_character(image_path='output/cont142.png'))


img = cv2.imread('fox2.png', cv2.IMREAD_COLOR_RGB)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

array = [gray,gray,gray,gray,gray]
for i in range(len(array)):
    array[i] = cv2.adaptiveThreshold(array[i],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,6.5)

erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
erosion_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))

for i in range(len(array)):
    array[i]= cv2.bitwise_not(array[i])
    array[i] = cv2.erode(array[i],erosion_kernel,iterations=1)
    array[i] = cv2.dilate(array[i],dilate_kernel,iterations=1)

horizontal_contours, hierarchy = cv2.findContours(array[4], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img=array[4]
iter=0
for cont in horizontal_contours:
    x,y,w,h = cv2.boundingRect(cont)
    if(w>500 or h>500) :
        continue
    #rect = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),5)
    crop = img[y:y+h,x:x+w]
    cv2.imwrite("./temp.png",crop,[cv2.IMWRITE_PNG_BILEVEL, 1])

    window_name = predict_character(image_path="./temp.png")
    #crop = cv2.erode(crop,erosion_kernel2,iterations=1)
    cv2.imshow(window_name, crop)
    cv2.waitKey(0)
    if(iter%5==0):
        cv2.destroyAllWindows()
    print(iter)
    iter=iter+1