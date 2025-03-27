import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
from src.visualization.charts import DataVisualizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Загружает все CSV файлы из указанной директории.
    
    :param data_dir: Путь к директории с данными
    :return: Словарь DataFrame с загруженными данными
    """
    data = {}
    
    # Проверка существования директории
    if not os.path.exists(data_dir):
        logger.error(f"Директория не существует: {data_dir}")
        return data
    
    # Проверка, что директория не пустая
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        logger.warning(f"В директории {data_dir} нет CSV файлов")
        return data
    
    # Загрузка CSV файлов
    for file in files:
        try:
            name = os.path.splitext(file)[0]
            filepath = os.path.join(data_dir, file)
            
            # Расширенная проверка файла
            if not os.path.isfile(filepath):
                logger.warning(f"Пропуск {file}: не является файлом")
                continue
            
            if os.path.getsize(filepath) == 0:
                logger.warning(f"Пропуск {file}: пустой файл")
                continue
            
            # Загрузка CSV с дополнительными параметрами
            df = pd.read_csv(
                filepath, 
                encoding='utf-8',  # Явное указание кодировки
                low_memory=False,  # Для больших файлов
                parse_dates=True   # Автоматический парсинг дат
            )
            
            # Проверка на пустой DataFrame
            if df.empty:
                logger.warning(f"Пропуск {file}: DataFrame пуст после загрузки")
                continue
            
            data[name] = df
            logger.info(f"Успешно загружен файл: {file}")
        
        except pd.errors.EmptyDataError:
            logger.error(f"Ошибка чтения {file}: файл пуст")
        except pd.errors.ParserError:
            logger.error(f"Ошибка парсинга {file}: некорректный формат CSV")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке {file}: {str(e)}")
    
    return data

def save_plot(
    fig: Optional[plt.Figure], 
    filename: str, 
    output_dir: str = 'reports/visualizations'
) -> None:
    """
    Сохраняет график в файл с расширенной обработкой ошибок.
    
    :param fig: Объект графика matplotlib
    :param filename: Имя файла для сохранения
    :param output_dir: Директория для сохранения
    """
    try:
        # Создание директории с полными правами
        os.makedirs(output_dir, exist_ok=True)
        
        # Полный путь к файлу
        full_path = os.path.join(output_dir, filename)
        
        # Проверка графика перед сохранением
        if fig is None:
            logger.warning(f"Пропуск сохранения {filename}: график не создан")
            return
        
        # Сохранение с высоким разрешением
        fig.savefig(
            full_path, 
            dpi=300,  # Высокое разрешение
            bbox_inches='tight'  # Обрезка лишних полей
        )
        logger.info(f"Сохранен график: {filename}")
    
    except Exception as e:
        logger.error(f"Ошибка сохранения графика {filename}: {str(e)}")
    finally:
        # Всегда закрываем график
        plt.close(fig)

def validate_dataframe(
    df: pd.DataFrame, 
    required_columns: list = None
) -> bool:
    """
    Валидация DataFrame перед визуализацией.
    
    :param df: DataFrame для проверки
    :param required_columns: Список обязательных колонок
    :return: True, если DataFrame корректен
    """
    if df is None or df.empty:
        logger.warning("DataFrame пуст или равен None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Отсутствуют колонки: {missing_columns}")
            return False
    
    return True

def main():
    """Основная функция обработки и визуализации данных."""
    try:
        # Директория с данными
        data_dir = 'data/processed'
        
        # Загрузка данных
        data = load_data(data_dir)
        
        # Создание визуализатора
        visualizer = DataVisualizer(pd.DataFrame())
        
        # Графики для различных наборов данных
        visualization_tasks = [
            ('sentiment_analysis', 
             visualizer.plot_sentiment_distribution, 
             'sentiment_distribution.png', 
             None),
            ('keywords', 
             visualizer.plot_top_keywords, 
             'top_keywords.png', 
             ['keyword', 'count']),
            ('message_trends', 
             visualizer.plot_trends, 
             'message_trends.png', 
             ['date', 'value']),
            ('theme_classification', 
             visualizer.plot_clusters, 
             'theme_clusters.png', 
             ['x', 'y', 'cluster'])
        ]
        
        # Выполнение визуализации
        for dataset_name, plot_func, filename, required_columns in visualization_tasks:
            if dataset_name in data:
                df = data[dataset_name]
                
                if validate_dataframe(df, required_columns):
                    try:
                        fig = plot_func(df) if required_columns else plot_func()
                        save_plot(fig, filename)
                    except Exception as e:
                        logger.error(f"Ошибка создания графика {filename}: {str(e)}")
    
    except Exception as e:
        logger.critical(f"Критическая ошибка в основной функции: {str(e)}")

if __name__ == '__main__':
    main()