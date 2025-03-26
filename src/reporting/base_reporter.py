import os
import pandas as pd
from abc import ABC, abstractmethod

class BaseReporter(ABC):
    """
    Абстрактный базовый класс для генерации отчетов.
    """
    
    def __init__(self, input_data_path="data/processed", output_dir="reports"):
        self.input_data_path = input_data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def generate_report(self):
        """
        Метод, который должен быть реализован в подклассах для генерации отчета.
        """
        pass
    
    def load_data(self, filename):
        """
        Загружает CSV-файл в DataFrame.
        """
        file_path = os.path.join(self.input_data_path, filename)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"Файл {file_path} не найден.")
    
    def save_report(self, content, filename):
        """
        Сохраняет отчет в текстовый файл.
        """
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Отчет сохранен: {file_path}")