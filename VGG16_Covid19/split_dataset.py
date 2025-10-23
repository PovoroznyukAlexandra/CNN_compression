import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets
import zipfile


def split_dataset_to_folders(source_dir, train_dir, test_dir, test_size=0.2, random_state=42):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    dataset = datasets.ImageFolder(root=source_dir)
    class_names = dataset.classes
    class_to_idx = dataset.class_to_idx
    
    print(f"Исходный датасет: {len(dataset)} изображений")
    print(f"Классы: {class_names}")
    
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    for class_name in class_names:
        class_path = os.path.join(source_dir, class_name)
        
        image_files = [f for f in os.listdir(class_path) 
                      if os.path.isfile(os.path.join(class_path, f))]
        
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Класс {class_name}: {len(image_files)} изображений")
        
        if len(image_files) == 0:
            continue
        
        train_files, test_files = train_test_split(
            image_files,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        for file_name in train_files:
            src_path = os.path.join(class_path, file_name)
            dst_path = os.path.join(train_dir, class_name, file_name)
            shutil.copy2(src_path, dst_path)
        
        for file_name in test_files:
            src_path = os.path.join(class_path, file_name)
            dst_path = os.path.join(test_dir, class_name, file_name)
            shutil.copy2(src_path, dst_path)
        
        print(f"Train: {len(train_files)}, Test: {len(test_files)}")

    save_split_info(source_dir, train_dir, test_dir, class_names)

def save_split_info(source_dir, train_dir, test_dir, class_names):
    info_file = "split_dataset/split_info.txt"
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("=== ИНФОРМАЦИЯ О РАЗДЕЛЕНИИ ДАТАСЕТА ===\n\n")
        f.write(f"Исходная папка: {source_dir}\n")
        f.write(f"Train папка: {train_dir}\n")
        f.write(f"Test папка: {test_dir}\n\n")
        
        f.write("РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ:\n")
        f.write("Класс         | Train | Test  | Всего\n")
        f.write("-" * 40 + "\n")
        
        total_train = 0
        total_test = 0
        total_all = 0
        
        for class_name in class_names:
            train_count = len(os.listdir(os.path.join(train_dir, class_name)))
            test_count = len(os.listdir(os.path.join(test_dir, class_name)))
            source_count = len(os.listdir(os.path.join(source_dir, class_name)))
            
            total_train += train_count
            total_test += test_count
            total_all += source_count
            
            f.write(f"{class_name:12} | {train_count:5} | {test_count:5} | {source_count:6}\n")
        
        f.write("-" * 40 + "\n")
        f.write(f"{'Всего':12} | {total_train:5} | {total_test:5} | {total_all:6}\n")
        f.write(f"Соотношение: {total_train/total_all:.1%} train / {total_test/total_all:.1%} test\n")

if __name__ == "__main__":
    with zipfile.ZipFile('image_dataset.zip', 'r') as zip_file:
        zip_file.extractall('image_dataset')

    source_dataset = "./image_dataset"
    train_dataset = "./split_dataset/dataset_train"
    test_dataset = "./split_dataset/dataset_test"
    test_size = 0.2
    random_state = 42

    print(f"Источник: {source_dataset}")
    print(f"Train: {train_dataset}")
    print(f"Test: {test_dataset}")
    print(f"Test size: {test_size}")
    print(f"Random state: {random_state}")
    print("-" * 50)
    
    split_dataset_to_folders(
        source_dir=source_dataset,
        train_dir=train_dataset,
        test_dir=test_dataset,
        test_size=test_size,
        random_state=random_state
    )

    dataset = datasets.ImageFolder(root=source_dataset)
    class_names = dataset.classes

    print(f"Информация сохранена в: {os.path.join(os.path.dirname(train_dataset), 'split_dataset/split_info.txt')}")