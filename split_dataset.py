import os
import random
import shutil
 
source = r"C:\fruits\Tomatoes"

train_dir = r"C:\VGG16-Fruit-Recognition\dataset\train\Tomatoes"
val_dir = r"C:\VGG16-Fruit-Recognition\dataset\val\Tomatoes"
test_dir = r"C:\VGG16-Fruit-Recognition\dataset\test\Tomatoes"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

images = os.listdir(source)
random.shuffle(images)

total = len(images)

train_count = int(total * 0.70)
val_count = int(total * 0.15)

train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

for img in train_images:
    shutil.copy(os.path.join(source, img),
                os.path.join(train_dir, img))

for img in val_images:
    shutil.copy(os.path.join(source, img),
                os.path.join(val_dir, img))

for img in test_images:
    shutil.copy(os.path.join(source, img),
                os.path.join(test_dir, img))

print("Train:", len(train_images))
print("Validation:", len(val_images))
print("Test:", len(test_images))