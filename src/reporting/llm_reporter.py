import openai
from config.llm_config import OPENAI_API_KEY
from src.utils.logger import Logger

openai.api_key = OPENAI_API_KEY

class LLMReporter:
    def __init__(self):
        self.logger = Logger("LLMReporter")

    def generate(self, df, time_range=None):
        # Агрегация данных для отчёта
        sentiment_counts = df["sentiment_category"].value_counts().to_dict()
        sentiment_stats = df["sentiment_score"].describe()
        top_keywords = df["text"].str.split().explode().value_counts().head(10)
        msg_length_stats = df["text"].str.len().describe()
        
        # Формирование детального промпта
        prompt = f"""
        Составь детализированный аналитический отчёт на основе данных:
        {f"Период анализа: {time_range}" if time_range else ""}
        
        Общая статистика:
        - Количество сообщений: {len(df)}
        - Средняя длина сообщения: {msg_length_stats['mean']:.1f} символов
        
        Анализ тональности:
        - Распределение: {sentiment_counts}
        - Средний показатель сентимента: {sentiment_stats['mean']:.2f}
        - Максимальный позитив: {sentiment_stats['max']:.2f}
        - Максимальный негатив: {sentiment_stats['min']:.2f}
        
        Ключевые слова:
        - Топ-10 ключевых слов: {top_keywords.index.tolist()}
        - Частота использования:
        {top_keywords.to_dict()}
        
        Примеры сообщений:
        {df['text'].sample(3).tolist()}
        
        Сделай выводы и рекомендации на основе анализа.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Ошибка генерации отчёта: {e}")
            return "Ошибка при генерации отчёта"
