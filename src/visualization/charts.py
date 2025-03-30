import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DataVisualizer:
    """
    Класс для визуализации данных: создание графиков для анализа.
    Предоставляет методы для визуализации различных аспектов данных.
    """

    def __init__(self, data):
        """
        Инициализация визуализатора с проверкой входных данных.
        
        :param data: Pandas DataFrame с исходными данными
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Входные данные должны быть DataFrame")
        self.data = data

    def _create_figure(self, figsize=(10, 6)):
        """
        Создает стандартизированную фигуру для графиков.
        
        :param figsize: Размер фигуры (ширина, высота)
        :return: Объект фигуры matplotlib
        """
        plt.close('all')  # Закрытие всех предыдущих графиков
        return plt.figure(figsize=figsize, dpi=100)

    def plot_sentiment_distribution(self):
        """
        Создает график распределения тональности сообщений.
        
        :return: Фигура matplotlib с графиком распределения или None если данные отсутствуют
        """
        try:
            if 'category' not in self.data.columns:
                print("Пропуск графика тональности - столбец 'category' отсутствует")
                return None
                
            sentiment_counts = self.data['category'].value_counts()
            
            # Convert index to string to ensure categorical plotting
            sentiment_counts.index = sentiment_counts.index.astype(str)
            
            fig = self._create_figure(figsize=(8, 6))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, 
                       hue=sentiment_counts.index, palette='viridis', legend=False)
            
            plt.title('Распределение тональности сообщений', fontsize=14, fontweight='bold')
            plt.xlabel('Категория тональности', fontsize=10)
            plt.ylabel('Количество сообщений', fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Ошибка в plot_sentiment_distribution: {e}")
            return None

    def plot_top_keywords(self, keywords_df, top_n=10):
        """
        Строит график с наибольшими по частоте упоминаниями ключевыми словами.
        
        :param keywords_df: DataFrame с ключевыми словами
        :param top_n: Количество топовых ключевых слов
        :return: Фигура matplotlib с графиком ключевых слов
        """
        try:
            top_keywords = keywords_df.head(top_n)
            
            fig = self._create_figure(figsize=(10, 6))
            sns.barplot(x='count', y='keyword', data=top_keywords, hue='keyword', palette='Blues_d', legend=False)
            
            plt.title(f'Top {top_n} часто упоминаемых ключевых слов', fontsize=14, fontweight='bold')
            plt.xlabel('Частота', fontsize=10)
            plt.ylabel('Ключевые слова', fontsize=10)
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Ошибка в plot_top_keywords: {e}")
            fig = self._create_figure()
            plt.text(0.5, 0.5, f'Ошибка визуализации: {e}', 
                     horizontalalignment='center', verticalalignment='center')
            return fig

    def plot_trends(self, trend_data):
        """
        Строит график тренда сообщений по времени.
        
        :param trend_data: DataFrame с трендовыми данными
        :return: Фигура matplotlib с графиком трендов
        """
        try:
            fig = self._create_figure(figsize=(10, 6))
            
            if not isinstance(trend_data, pd.DataFrame):
                raise ValueError("trend_data должен быть pandas DataFrame")
            
            # Проверка колонок для линейного графика
            required_columns = ['date', '0']
            for col in required_columns:
                if col not in trend_data.columns:
                    raise ValueError(f"Отсутствует обязательная колонка: {col}")
            
            # Ensure date column is datetime
            trend_data = trend_data.copy()
            trend_data['date'] = pd.to_datetime(trend_data['date'])
            
            sns.lineplot(x='date', y='0', data=trend_data, color='#1f77b4', linewidth=2)
            
            plt.title('Тренды сообщений по времени', fontsize=14, fontweight='bold')
            plt.xlabel('Дата', fontsize=10)
            plt.ylabel('Количество сообщений', fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Ошибка в plot_trends: {e}")
            fig = self._create_figure()
            plt.text(0.5, 0.5, f'Ошибка визуализации: {e}', 
                     horizontalalignment='center', verticalalignment='center')
            return fig

    def plot_clusters(self, df, n_clusters=5):
        """
        Строит график кластеризации сообщений.
        
        :param df: DataFrame с координатами и кластерами
        :param n_clusters: Количество кластеров
        :return: Фигура matplotlib с графиком кластеров или None если данные отсутствуют
        """
        try:
            # Проверка обязательных колонок
            required_columns = ['x', 'y', 'cluster']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Пропуск графика кластеров - отсутствуют колонки: {missing_cols}")
                return None
            
            fig = self._create_figure(figsize=(8, 6))
            sns.scatterplot(data=df, x='x', y='y', hue='cluster', 
                            palette='deep', s=100, alpha=0.7)
            
            plt.title(f'Кластеры сообщений (n_clusters={n_clusters})', 
                      fontsize=14, fontweight='bold')
            plt.xlabel('X', fontsize=10)
            plt.ylabel('Y', fontsize=10)
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Ошибка в plot_clusters: {e}")
            return None

    def plot_sentiment_by_time(self, sentiment_data):
        """
        Строит график распределения тональности по времени суток.
        
        :param sentiment_data: DataFrame с данными о тональности и времени
        :return: Фигура matplotlib с графиком распределения
        """
        try:
            if not isinstance(sentiment_data, pd.DataFrame):
                raise ValueError("sentiment_data должен быть pandas DataFrame")
            
            # Преобразование даты и группировка
            sentiment_data = sentiment_data.copy()
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            sentiment_data['hour'] = sentiment_data['date'].dt.hour
            sentiment_hourly = sentiment_data.groupby('hour')['category'].value_counts().unstack().fillna(0)
            
            fig = self._create_figure(figsize=(12, 6))
            
            sentiment_hourly.plot(kind='bar', stacked=True)
            
            plt.title('Распределение тональности по времени суток', 
                      fontsize=14, fontweight='bold')
            plt.xlabel('Час', fontsize=10)
            plt.ylabel('Количество сообщений', fontsize=10)
            plt.xticks(rotation=0)
            plt.legend(title='Тональность', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Ошибка в plot_sentiment_by_time: {e}")
            fig = self._create_figure()
            plt.text(0.5, 0.5, f'Ошибка обработки данных: {e}', 
                     horizontalalignment='center', verticalalignment='center')
            return fig
