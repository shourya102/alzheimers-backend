from keras.models import load_model
from f1_score import F1Score
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_addons as tf

model = load_model('model/efficientnet_b0_alz.h5', custom_objects={'F1Score': tf.metrics.F1Score})

test_dataset = image_dataset_from_directory(
    'dataset/alzheimers-dataset/test',
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

X_test = []
y_test = []

for images, labels in test_dataset:
    X_test.extend(images.numpy())
    y_test.extend(labels.numpy())

X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=-1)

accuracy = accuracy_score(y_test, y_pred_classes)
report = classification_report(y_test, y_pred_classes, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report_efficientnet.csv', index=True)
print(report_df)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.class_names, yticklabels=test_dataset.class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
