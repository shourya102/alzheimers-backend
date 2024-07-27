import os

import tensorflow_addons
from keras import Model, Sequential
from keras.applications import EfficientNetV2B0
from keras.applications.resnet import ResNet50
from keras.applications.vgg19 import VGG19
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import GlobalAveragePooling2D, Flatten, Dropout
from keras.layers.core import Dense
from keras.metrics import Precision, Recall
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from f1_score import F1Score
from img_prediction import plot_metrics

BATCH_SIZE = 32


def generator(path):
    classes = os.listdir(path)
    batches = ImageDataGenerator().flow_from_directory(path, target_size=(224, 224), classes=classes,
                                                       batch_size=BATCH_SIZE, class_mode='categorical')
    return batches


def fine_tune(base_model, train_path, val_path, model_name, loss, classes, epochs):
    train_generator = generator(train_path)
    val_generator = generator(val_path)
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(len(classes), activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer=Adam(),
                  loss=loss,
                  metrics=['accuracy', Precision(), Recall(), F1Score()])
    history = model.fit(train_generator,
                        epochs=epochs,
                        steps_per_epoch=int(train_generator.samples / BATCH_SIZE),
                        validation_data=val_generator,
                        validation_steps=int(val_generator.samples / BATCH_SIZE)
                        )
    plot_metrics(history)
    model.save('model/' + model_name, save_format='tf')
    print(f'Model saved at {model_name}')


def fine_tune_effnet(train_path, val_path, model_name, loss, classes, epochs):
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(classes), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=loss,
        metrics=['accuracy', Precision(), Recall(),
                 tensorflow_addons.metrics.F1Score(num_classes=len(classes), average='macro')]
    )
    model.summary()
    train_generator = generator(train_path)
    val_generator = generator(val_path)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[reduce_lr, early_stopping]
    )
    plot_metrics(history)
    model.save('model/' + model_name)
    print(f'Model saved at model/{model_name}')
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=loss,
        metrics=['accuracy', Precision(), Recall(),
                 tensorflow_addons.metrics.F1Score(num_classes=len(classes), average='macro')]
    )
    history = model.fit(
        train_generator,
        epochs=epochs // 2,
        validation_data=val_generator,
        callbacks=[reduce_lr, early_stopping]
    )
    plot_metrics(history)
    model.save('model/' + model_name)
    print(f'Final model saved at model/{model_name}')


if __name__ == '__main__':
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model1 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    train_path = 'dataset/alzheimers-dataset/train'
    train_path1 = 'dataset/alzheimers-dataset/preprocessed_train'
    val_path = 'dataset/alzheimers-dataset/val'
    val_path1 = 'dataset/alzheimers-dataset/preprocessed_val'
    model_name = 'resnet50_alz.h5'
    model_name1 = 'vgg19_alz.h5'
    model_name3 = 'efficientnet_b0_alz.h5'
    loss = 'categorical_crossentropy'
    classes = os.listdir(train_path)
    epochs = 25
    fine_tune_effnet(train_path1, val_path1, model_name3, loss, classes, epochs)
