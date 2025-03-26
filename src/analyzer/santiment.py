from textblob import TextBlob
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentAnalyzer:
    """
    Класс для анализа тональности текста с использованием TextBlob.
    """

    def __init__(self, language='russian'):
        self.language = language
        self.sentiments = []

    def analyze_sentiment(self, text):
        """
        Анализирует текст и возвращает оценку тональности и категорию.
        """
        try:
            blob = TextBlob(text)
            score = blob.sentiment.polarity  # Оценка тональности от -1 (негатив) до 1 (позитив)
            category = self._categorize_sentiment(score)
            return score, category
        except Exception as e:
            print(f"❌ Ошибка при анализе текста: {e}")
            return None, None

    def _categorize_sentiment(self, score):
        """
        Категоризация тональности на основе оценки.
        """
        if score > 0.2:
            return 'позитив'
        elif score < -0.2:
            return 'негатив'
        else:
            return 'нейтрально'

    def analyze_messages_from_csv(self, df=None, input_path="data/raw/messages.csv", output_path="data/processed/sentiment_analysis.csv"):
        """Анализирует сообщения из CSV файла или DataFrame и сохраняет результаты"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if df is None:
                df = pd.read_csv(input_path)
            
            if 'text' not in df.columns:
                raise ValueError("Данные должны содержать колонку 'text'")
            
            results = []
            for text in df['text'].dropna():
                score, category = self.analyze_sentiment(text)
                results.append({
                    'text': text,
                    'score': score,
                    'category': category
                })
            
            result_df = pd.DataFrame(results)
            
            # Add date if present in input
            if 'date' in df.columns:
                result_df['date'] = df['date'].iloc[:len(result_df)]
            
            result_df.to_csv(output_path, index=False)
            print(f"✅ Анализ тональности сохранен в {output_path}")
            return result_df
            
        except Exception as e:
            print(f"❌ Ошибка анализа тональности: {e}")
            raise

    def analyze_messages(self, messages):
        """Анализирует список сообщений напрямую без сохранения в файл"""
        if not isinstance(messages, (list, pd.Series)):
            raise ValueError("messages должен быть списком или pandas Series")
            
        results = []
        for text in messages:
            if pd.isna(text):
                continue
            score, category = self.analyze_sentiment(text)
            results.append({
                'text': text,
                'score': score,
                'category': category
            })
        return pd.DataFrame(results)

    def generate_sentiment_report(self, output_path="data/processed/sentiment_report.csv"):
        """
        Генерирует отчет по тональности и сохраняет в файл CSV.
        """
        if not self.sentiments:
            print("Нет данных для отчета.")
            return
        
        df = pd.DataFrame(self.sentiments)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Отчет по тональности сохранен в {output_path}")

    def visualize_sentiment(self):
        """
        Создает график распределения тональности.
        """
        if not self.sentiments:
            print("Нет данных для визуализации.")
            return
        
        sentiment_counts = pd.Series([entry['category'] for entry in self.sentiments]).value_counts()
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        plt.title('Распределение тональности сообщений')
        plt.xlabel('Категория тональности')
        plt.ylabel('Количество сообщений')
        plt.show()
