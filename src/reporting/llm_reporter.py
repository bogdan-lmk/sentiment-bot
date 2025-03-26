import openai
import os
import pandas as pd
from collections import Counter
from src.reporting.base_reporter import BaseReporter

class LLMReporter(BaseReporter):
    """
    Генерирует расширенный отчет с использованием модели OpenAI GPT 
    с глубоким анализом сообщений.
    """

    def __init__(self, input_data_path="data/processed", model="gpt-3.5-turbo", max_tokens=3000):
        super().__init__(input_data_path=input_data_path)
        self.model = model
        self.max_tokens = max_tokens
        openai.api_key = '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' 

    def _preprocess_messages(self, messages):
        """
        Подготовка и анализ сообщений перед отправкой в GPT.
        """
        # Удаление дубликатов и подсчет частоты сообщений
        message_counter = Counter(messages)
        
        # Сортировка сообщений по частоте
        most_common_messages = message_counter.most_common(20)
        
        return {
            'total_messages': len(messages),
            'unique_messages': len(set(messages)),
            'most_common_messages': most_common_messages
        }

    def generate_report(self):
        """
        Создает глубокий и детальный отчет с помощью GPT.
        """
        try:
            # Проверка на наличие файла перед загрузкой
            messages_file_path = os.path.join(self.input_data_path, "messages.csv")
            if not os.path.exists(messages_file_path):
                print(f"Файл {messages_file_path} не найден.")
                return "No messages data found"

            # Загружаем данные сообщений
            messages_df = self.load_data("messages.csv")
            messages = messages_df["text"].dropna().tolist()

            if not messages:
                print("Нет данных для анализа.")
                return "No messages to analyze"

            # Загружаем данные анализа тональности, если доступны
            sentiment_stats = {}
            sentiment_path = os.path.join(self.input_data_path, "sentiment_analysis.csv")
            if os.path.exists(sentiment_path):
                sentiment_df = pd.read_csv(sentiment_path)
                sentiment_stats = {
                    'positive': len(sentiment_df[sentiment_df['category'] == 'позитив']),
                    'negative': len(sentiment_df[sentiment_df['category'] == 'негатив']),
                    'neutral': len(sentiment_df[sentiment_df['category'] == 'нейтрально']),
                    'total': len(sentiment_df)
                }

            # Предварительная обработка сообщений
            message_stats = self._preprocess_messages(messages)

            # Формирование расширенного промпта
            prompt = f"""
            Выполни глубокий и всесторонний анализ сообщений со следующими параметрами:

            Общая статистика:
            - Общее количество сообщений: {message_stats['total_messages']}
            - Уникальных сообщений: {message_stats['unique_messages']}
            {f"- Позитивных сообщений: {sentiment_stats.get('positive', 'N/A')} ({round(sentiment_stats.get('positive', 0)/sentiment_stats.get('total', 1)*100)}%)" if sentiment_stats else ""}
            {f"- Негативных сообщений: {sentiment_stats.get('negative', 'N/A')} ({round(sentiment_stats.get('negative', 0)/sentiment_stats.get('total', 1)*100)}%)" if sentiment_stats else ""}
            {f"- Нейтральных сообщений: {sentiment_stats.get('neutral', 'N/A')} ({round(sentiment_stats.get('neutral', 0)/sentiment_stats.get('total', 1)*100)}%)" if sentiment_stats else ""}

            Наиболее частые сообщения:
            {chr(10).join([f"{msg} (встречается {count} раз)" for msg, count in message_stats['most_common_messages']])}

            Требования к анализу:
            1. Определи основные тематические кластеры сообщений
            2. Проанализируй эмоциональную окраску сообщений (используя предоставленную статистику тональности)
            3. Выдели ключевые проблемы и потребности
            4. Предоставь рекомендации на основе анализа тональности
            5. Используй структурированный формат с заголовками

            Для анализа используй первые 70 сообщений:
            {chr(10).join(messages[:70])}
            """

            # Отправка запроса в OpenAI
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты опытный аналитик данных, способный выполнять глубокий контент-анализ."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens
            )

            # Извлечение сгенерированного отчета
            report_content = response["choices"][0]["message"]["content"]

            # Сохранение отчета
            self.save_report(report_content, "detailed_llm_report.txt")
            print("Детальный отчет GPT успешно создан.")

            # Generate PDF version
            from src.reporting.pdf_reporter import PDFReporter
            pdf_reporter = PDFReporter()
            pdf_reporter.generate_report(report_content)
            print("PDF версия отчета создана.")

            return report_content

        except openai.error.OpenAIError as e:
            print(f"Ошибка при обращении к OpenAI API: {e}")
            return f"LLM report unavailable - API error: {e}"
        except Exception as e:
            print(f"Ошибка при генерации отчета через GPT: {e}")
            return f"LLM report unavailable - error: {e}"

def main():
    reporter = LLMReporter()
    report = reporter.generate_report()
    print(report)

if __name__ == "__main__":
    main()
