import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

img_size = 224
batch_size = 32 #images pet training step
data_dir = "Data"

train_ds = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, #convert pixel values from 0-255 to 0-1 (normalization)
    validation_split=0.1,   #10% validation
    rotation_range=15,  #data augmentation
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest' #filling empty spaces with nearest pixel values
).flow_from_directory(  #we load the dataset/images from disk
    data_dir,
    target_size=(img_size, img_size),   #resize images to 224x224 (which they already are, just to be safe)
    batch_size=batch_size,
    subset="training",
    class_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
).flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset="validation",
    class_mode="categorical",
    shuffle=False
)


base = tf.keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

num_classes = train_ds.num_classes
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.25),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=15
)



#Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



model.save("hand_model.h5")
print("Model saved.")
