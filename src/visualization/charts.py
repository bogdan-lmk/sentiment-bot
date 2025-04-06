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
    def __init__(self, data, active_geo=None):
        self.active_geo = active_geo
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
            if 'sentiment' not in self.data.columns:
                print("Пропуск графика тональности - столбец 'sentiment' отсутствует")
                return None

            sentiment_counts = self.data['sentiment'].value_counts()
            sentiment_counts.index = sentiment_counts.index.astype(str)
            total = sentiment_counts.sum()

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

    def plot_top_phrases(self, phrases_df, top_n=15):
        """
        Строит интерактивный график наиболее часто упоминающихся фраз с аннотациями.
        """
        try:
            # Ensure we have the expected columns
            if not all(col in phrases_df.columns for col in ['phrase', 'count']):
                print("DataFrame missing required columns ('phrase' and 'count')")
                return None
                
            top_phrases = phrases_df.sort_values('count', ascending=False).head(top_n)
            total = top_phrases['count'].sum()

            fig = self._create_figure(figsize=(12, 8))
            ax = sns.barplot(
                x=top_phrases['count'], 
                y=top_phrases['phrase'],
                palette='rocket', 
                orient='h'
            )
            
            # Add annotations
            for i, (phrase, count) in enumerate(zip(top_phrases['phrase'], top_phrases['count'])):
                ax.text(
                    count + 0.1, 
                    i,
                    f"{count} ({count/total:.1%})",
                    va='center', 
                    fontsize=10, 
                    color='navy'
                )
            
            plt.title(f'Топ {top_n} наиболее частых фраз', fontsize=16, pad=20)
            plt.xlabel('Частота упоминаний', fontsize=12)
            plt.ylabel('Фраза', fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Ошибка в plot_top_phrases: {e}")
            return None

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
            
            required_columns = ['date', '0']
            for col in required_columns:
                if col not in trend_data.columns:
                    raise ValueError(f"Отсутствует обязательная колонка: {col}")
            
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
            return None

    def plot_clusters(self, df, n_clusters=5):
        """
        Строит интерактивный график кластеризации с улучшенной визуализацией.
        """
        import logging
        logger = logging.getLogger(__name__)
        fig = self._create_figure(figsize=(10, 8))
        try:
            # Validate data and columns
            required_columns = ['x', 'y', 'cluster']
            if not all(col in df.columns for col in required_columns):
                logger.warning("Пропуск графика кластеров - отсутствуют необходимые колонки")
                plt.close(fig)
                return None  # Return None to indicate skip
            if df.empty or len(df['cluster'].unique()) < 2:
                logger.warning("Недостаточно данных для кластеризации")
                return fig

            ax = sns.scatterplot(data=df, x='x', y='y', hue='cluster',
                               palette='viridis', s=120, alpha=0.8,
                               edgecolor='w', linewidth=0.5)
            
            centers = df.groupby('cluster')[['x', 'y']].mean()
            ax.scatter(centers.x, centers.y, s=300, c='red', 
                      marker='X', edgecolor='black', label='Центры кластеров')
            
            plt.title(f'Кластеризация сообщений ({n_clusters} кластеров)', 
                     fontsize=14, pad=20)
            plt.xlabel('Компонента X', fontsize=12)
            plt.ylabel('Компонента Y', fontsize=12)
            plt.legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Ошибка в plot_clusters: {e}")
            return None

    def plot_needs_distribution(self, needs_data):
        """
        Создает график распределения потребностей.
        :param needs_data: DataFrame с колонкой 'needs'
        :return: Фигура matplotlib или None
        """
        try:
            if 'needs' not in needs_data.columns:
                print("Пропуск графика потребностей - отсутствует колонка 'needs'")
                return None

            # Extract and count individual needs
            all_needs = []
            for needs_list in needs_data['needs'].str.findall(r"'([^']*)'"):
                all_needs.extend(needs_list)
            
            needs_counts = pd.Series(all_needs).value_counts()
            
            fig = self._create_figure(figsize=(12, 6))
            sns.barplot(x=needs_counts.index, y=needs_counts.values,
                       hue=needs_counts.index, palette='viridis', legend=False)
            
            plt.title('Распределение потребностей', fontsize=14)
            plt.xlabel('Категория потребности', fontsize=10)
            plt.ylabel('Количество упоминаний', fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Ошибка в plot_needs_distribution: {e}")
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
            return None
