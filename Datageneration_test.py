import os
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
def prep_XY(folder_path, label, X, Y):
    D=read_and_convert_images(folder_path)
    label=tf.constant([label])
    for i in range(100):
        p=D[i].reshape(40000)
        print(p)
        label=np.array(label)
        print(label)
        X.append(p)
        Y.append(label)
def read_and_convert_images(folder_path, target_size=(200, 200)):
    image_tensors = []
    for i in range(100):
        image_file=os.listdir(folder_path)[i]
   # for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = load_img(image_path, target_size=target_size, color_mode='grayscale')  # Grayscale (single-channel) image
        # If RGB images, set color_mode='rgb' and target_size=(200, 200, 3)
        image_array = img_to_array(image)
        image_tensors.append(image_array)
    return image_tensors
X_train=[]
Y_train=[]
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\A', 1, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\B', 2, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\C', 3, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\D', 4, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\E', 5, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\F', 6, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\G', 7, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\H', 8, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\I', 9, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\J', 10, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\K', 11, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\L', 12, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\M', 13, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\N', 14, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\O', 15, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\P', 16, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\Q', 17, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\R', 18, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\S', 19, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\T', 20, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\U', 21, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\V', 22, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\W', 23, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\X', 24, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\Y', 25, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\Z', 26, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\del', 27, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\nothing', 28, X_train, Y_train)
prep_XY(r'D:\Projects\ASL Translation\Data\asl_alphabet_train\asl_alphabet_train\space', 29, X_train, Y_train)
import csv
train=np.random.choice(100,70)
cv=np.random.choice(100,30)
X_train_new=[]
Y_train_new=[]
X_cv=[]
Y_cv=[]
for i in train:
    X_train_new.append(X_train[i])
    Y_train_new.append(Y_train[i])

for i in cv:
    X_cv.append(X_train[i])
    Y_cv.append(Y_train[i])

# The name of the CSV file to be created
csv_file_name = 'X_train.csv'

# Writing the array to a CSV file
with open(csv_file_name, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Iterate through each row in the array and write it to the CSV file
    for row in X_train_new:
        csv_writer.writerow(row)

print(f'Array data has been successfully stored in "{csv_file_name}"')

csv_file_name = 'Y_train.csv'

# Writing the array to a CSV file
with open(csv_file_name, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Iterate through each row in the array and write it to the CSV file
    for row in Y_train_new:
        csv_writer.writerow(row)

print(f'Array data has been successfully stored in "{csv_file_name}"')

csv_file_name = 'X_cv.csv'

# Writing the array to a CSV file
with open(csv_file_name, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Iterate through each row in the array and write it to the CSV file
    for row in X_cv:
        csv_writer.writerow(row)
print(f'Array data has been successfully stored in "{csv_file_name}"')

csv_file_name = 'Y_cv.csv'

# Writing the array to a CSV file
with open(csv_file_name, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Iterate through each row in the array and write it to the CSV file
    for row in Y_cv:
        csv_writer.writerow(row)

print(f'Array data has been successfully stored in "{csv_file_name}"')