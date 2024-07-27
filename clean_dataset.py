import os
import random
import shutil


def get_unique_filename(destination, filename):
    base, extension = os.path.splitext(filename)
    counter = 1
    while os.path.exists(os.path.join(destination, filename)):
        filename = f"{base}_{counter}{extension}"
        counter += 1
    return filename


def reorganize_dataset(root_dir):
    categories = os.listdir(root_dir)
    for category in categories:
        category_path = os.path.join(root_dir, category)
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    file_path = os.path.join(root, file)
                    shutil.move(file_path, os.path.join(category_path, get_unique_filename(category, file)))
        for root, dirs, files in os.walk(category_path, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)


def split_dataset(root_dir, train_ratio=0.8, val_ratio=0.1):
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    classes = os.listdir(root_dir)
    for category in classes:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)
        category_path = os.path.join(root_dir, category)
        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        random.shuffle(files)
        split_index_train = int(len(files) * train_ratio)
        train_files = files[:split_index_train]
        temp_files = files[split_index_train:]
        split_index_val = int(len(temp_files) * val_ratio)
        val_files = temp_files[:split_index_val]
        test_files = temp_files[split_index_val:]
        for file in train_files:
            src = os.path.join(category_path, file)
            dst = os.path.join(train_dir, category, file)
            shutil.copy2(src, dst)
        for file in val_files:
            src = os.path.join(category_path, file)
            dst = os.path.join(val_dir, category, file)
            shutil.copy2(src, dst)
        for file in test_files:
            src = os.path.join(category_path, file)
            dst = os.path.join(test_dir, category, file)
            shutil.copy2(src, dst)
    print(
        f"Dataset split completed. Train set: {len(train_files)}, Validation set: {len(val_files)}, Test set: {len(test_files)}")


if __name__ == '__main__':
    # reorganize_dataset(root_dir='datasets/brain_disease_dataset')
    split_dataset('dataset/alzheimers-dataset', train_ratio=0.8, val_ratio=0.5)
