import os
import shutil

class FileManager:
    """
    Класс для управления файлами: создание папок, перемещение и удаление файлов.
    """

    def __init__(self, base_dir="data/processed"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def move_file(self, source_path, dest_path):
        """
        Перемещает файл из source_path в dest_path.
        """
        if not os.path.exists(source_path):
            print(f"Исходный файл не найден: {source_path}")
            return

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(source_path, dest_path)
        print(f"Файл перемещен из {source_path} в {dest_path}")

    def delete_file(self, file_path):
        """
        Удаляет файл.
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Файл {file_path} удален.")
        else:
            print(f"Файл не найден: {file_path}")

    def list_files(self, directory=None):
        """
        Возвращает список файлов в указанной директории.
        Если директория не указана, используется base_dir.
        """
        directory = directory or self.base_dir
        if not os.path.exists(directory):
            print(f"Директория не найдена: {directory}")
            return []
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def get_file_path(self, filename):
        """
        Возвращает полный путь к файлу по имени.
        """
        return os.path.join(self.base_dir, filename)