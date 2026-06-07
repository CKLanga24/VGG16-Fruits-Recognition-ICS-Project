import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

train_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    r"C:\VGG16-Fruit-Recognition\dataset\train",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

print("Classes:", train_data.class_indices)

num_classes = len(train_data.class_indices)

val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    r"C:\VGG16-Fruit-Recognition\dataset\val",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    r"C:\VGG16-Fruit-Recognition\dataset\test",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

def custom_loss(y_true, y_pred):
    epsilon = 0.1
    num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)

    y_true_smooth = (
        (1.0 - epsilon) * y_true
        + (epsilon / num_classes)
    )

    gamma = 2.0

    y_pred_clipped = tf.clip_by_value(
        y_pred,
        1e-7,
        1.0 - 1e-7
    )

    cross_entropy = -tf.reduce_sum(
        y_true_smooth * tf.math.log(y_pred_clipped),
        axis=-1
    )

    focal_weight = tf.reduce_sum(
        y_true * tf.math.pow(
            1.0 - y_pred_clipped,
            gamma
        ),
        axis=-1
    )

    focal_loss = focal_weight * cross_entropy
    focal_loss = tf.reduce_mean(focal_loss)

    lambda_l2 = 0.01

    l2_penalty = lambda_l2 * tf.reduce_mean(
        tf.square(y_pred)
    )

    return focal_loss + l2_penalty

model.compile(
    optimizer="adam",
    loss=custom_loss,
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

test_loss, test_acc = model.evaluate(test_data)

print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

model.save("fruit_vgg16_model.h5")