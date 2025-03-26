import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    """
    Класс для визуализации данных: создание графиков для анализа.
    """

    def __init__(self, data):
        self.data = data

    def plot_sentiment_distribution(self):
        """
        Создает график распределения тональности сообщений.
        """
        sentiment_counts = self.data['category'].value_counts()

        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
        plt.title('Распределение тональности сообщений')
        plt.xlabel('Категория тональности')
        plt.ylabel('Количество сообщений')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()  # Return figure instead of showing

    def plot_top_keywords(self, keywords_df, top_n=10):
        """
        Строит график с наибольшими по частоте упоминаниями ключевыми словами.
        """
        top_keywords = keywords_df.head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='count', y='keyword', data=top_keywords, palette='Blues_d')
        plt.title(f'Top {top_n} часто упоминаемых ключевых слов')
        plt.xlabel('Частота')
        plt.ylabel('Ключевые слова')
        plt.tight_layout()
        return plt.gcf()  # Return figure instead of showing

    def plot_trends(self, trend_data):
        """
        Строит график тренда сообщений по времени.
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=trend_data, palette='coolwarm', linewidth=2)
        plt.title('Тренды сообщений по времени')
        plt.xlabel('Дата')
        plt.ylabel('Количество сообщений')
        plt.tight_layout()
        return plt.gcf()  # Return figure instead of showing

    def plot_clusters(self, df, n_clusters=5):
        """
        Строит график кластеризации сообщений.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='deep', s=100, alpha=0.7)
        plt.title(f'Кластеры сообщений (n_clusters={n_clusters})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        return plt.gcf()  # Return figure instead of showing

    def plot_sentiment_by_time(self, sentiment_data):
        """
        Строит график распределения тональности по времени.
        """
        if 'date' not in sentiment_data.columns:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No date data available', 
                    ha='center', va='center', fontsize=12)
            plt.title('Распределение тональности по времени суток')
            return plt.gcf()

        try:
            sentiment_data['hour'] = sentiment_data['date'].dt.hour
            sentiment_hourly = sentiment_data.groupby('hour')['category'].value_counts().unstack().fillna(0)

            plt.figure(figsize=(10, 6))
            sentiment_hourly.plot(kind='bar', stacked=True, figsize=(12, 6))
            plt.title('Распределение тональности по времени суток')
            plt.xlabel('Час')
            plt.ylabel('Количество сообщений')
            plt.xticks(rotation=0)
            plt.tight_layout()
            return plt.gcf()  # Return figure instead of showing
        except Exception as e:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Error processing date data: {str(e)}', 
                    ha='center', va='center', fontsize=12)
            plt.title('Распределение тональности по времени суток')
            return plt.gcf()
