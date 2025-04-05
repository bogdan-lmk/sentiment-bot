import os
import pandas as pd
from collections import Counter
import sys
import os



from src.reporting.base_reporter import BaseReporter
from dotenv import load_dotenv
import re
from typing import Optional
from src.reporting.llm_providers.openai_provider import OpenAIProvider
from src.reporting.llm_providers.deepseek_provider import DeepSeekProvider

load_dotenv()

class LLMReporter(BaseReporter):
    """
    Генерирует расширенный отчет с использованием модели OpenAI GPT 
    с глубоким анализом сообщений, включающий дополнительные аналитические insights.
    """

    def __init__(
        self,
        input_data_path="data/processed",
        output_dir="reports",
        provider="deepseek",  # Default to DeepSeek with OpenAI as fallback
        model="deepseek-reasoner",
        max_tokens=4000,
        active_geo: Optional[str] = None
    ):
        super().__init__(input_data_path=input_data_path, output_dir=output_dir)
        self.active_geo = active_geo
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize the selected LLM provider
        if provider == "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self.provider = OpenAIProvider(api_key, model)
        elif provider == "deepseek":
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is required")
            self.provider = DeepSeekProvider(api_key, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _preprocess_messages(self, messages):
        """
        Расширенная подготовка и анализ сообщений перед отправкой в GPT.
        """
        # Удаление дубликатов и подсчет частоты сообщений
        message_counter = Counter(messages)
        
        # Сортировка сообщений по частоте
        most_common_messages = message_counter.most_common(20)
        
        # Извлечение географических данных
        geography = self._extract_geography(messages)
        
        # Извлечение источников
        sources = self._extract_sources(messages)
        
        return {
            'total_messages': len(messages),
            'unique_messages': len(set(messages)),
            'most_common_messages': most_common_messages,
            'geography': geography,
            'sources': sources
        }

    def _extract_geography(self, messages):
        """
        Извлечение географических упоминаний из сообщений.
        """
        geography_pattern = r'\b(город|область|район|регион|страна|село|деревня)\s+([А-Яа-я\w-]+)'
        geo_mentions = {}
        
        for message in messages:
            matches = re.findall(geography_pattern, message, re.IGNORECASE)
            for match in matches:
                location = match[1]
                geo_mentions[location] = geo_mentions.get(location, 0) + 1
        
        return dict(sorted(geo_mentions.items(), key=lambda x: x[1], reverse=True)[:10])

    def _extract_sources(self, messages):
        """
        Извлечение источников информации из сообщений.
        """
        source_pattern = r'\b(из|от|источник|сообщает|согласно)\s+([А-Яа-я\w-]+)'
        source_mentions = {}
        
        for message in messages:
            matches = re.findall(source_pattern, message, re.IGNORECASE)
            for match in matches:
                source = match[1]
                source_mentions[source] = source_mentions.get(source, 0) + 1
        
        return dict(sorted(source_mentions.items(), key=lambda x: x[1], reverse=True)[:10])

    def generate_report(self):
        """
        Создаем информативный детальный отчет с помощью LLM с расширенными возможностями.
        """
        try:
            # Проверка на наличие файла перед загрузкой
            messages_file_path = os.path.join(self.input_data_path, self.active_geo, "processed_messages.csv")
            if not os.path.exists(messages_file_path):
                print(f"Файл {messages_file_path} не найден.")
                return "No messages data found"

            # Загружаем данные сообщений
            messages_df = pd.read_csv(messages_file_path)
            messages = messages_df["text"].dropna().tolist()

            if not messages:
                print("Нет данных для анализа.")
                return "No messages to analyze"

            # Загружаем данные анализа тональности, если доступны
            sentiment_stats = {}
            sentiment_path = os.path.join(self.input_data_path, self.active_geo, "sentiment_analysis.csv")
            if os.path.exists(sentiment_path):
                sentiment_df = pd.read_csv(sentiment_path)
                sentiment_stats = {
                    'positive': len(sentiment_df[sentiment_df['category'] == 'позитив']),
                    'negative': len(sentiment_df[sentiment_df['category'] == 'негатив']),
                    'neutral': len(sentiment_df[sentiment_df['category'] == 'нейтрально']),
                    'total': len(sentiment_df)
                }

            # Загружаем топ фраз, если доступны
            phrases_stats = []
            phrases_path = os.path.join(self.input_data_path, self.active_geo, "top_phrases.csv")
            if os.path.exists(phrases_path):
                phrases_df = pd.read_csv(phrases_path)
                phrases_stats = phrases_df.to_dict('records')

            # Предварительная обработка сообщений
            message_stats = self._preprocess_messages(messages)

            # Формирование расширенного промпта
            prompt = f"""
            Выполни глубокий и всесторонний анализ сообщений со следующими параметрами:

            Общая статистика:
            - Общее количество сообщений: {message_stats['total_messages']}
            - Уникальных сообщений: {message_stats['unique_messages']}
            {f"- Позитивных сообщений: {sentiment_stats.get('positive', 'N/A')} ({round(sentiment_stats.get('positive', 0)/sentiment_stats.get('total', 1)*100, 2)}%)" if sentiment_stats else ""}
            {f"- Негативных сообщений: {sentiment_stats.get('negative', 'N/A')} ({round(sentiment_stats.get('negative', 0)/sentiment_stats.get('total', 1)*100, 2)}%)" if sentiment_stats else ""}
            {f"- Нейтральных сообщений: {sentiment_stats.get('neutral', 'N/A')} ({round(sentiment_stats.get('neutral', 0)/sentiment_stats.get('total', 1)*100, 2)}%)" if sentiment_stats else ""}

            Наиболее частые сообщения:
            {chr(10).join([f"{msg} (встречается {count} раз)" for msg, count in message_stats['most_common_messages']])}

            Наиболее частые фразы (2-4 слова):
            {chr(10).join([f"{p['phrase']} (встречается {p['count']} раз)" for p in phrases_stats[:20]]) if phrases_stats else "Данные о фразах недоступны"}

            Географический срез:
            {chr(10).join([f"{location}: {count} упоминаний" for location, count in message_stats['geography'].items()])}

            Источники информации:
            {chr(10).join([f"{source}: {count} упоминаний" for source, count in message_stats['sources'].items()])}

            Требования к анализу:

            Определи основные тематические кластеры сообщений, выявляя ключевые темы и группы по схожим идеям.
            Проанализируй эмоциональную окраску сообщений, используя предоставленную статистику тональности для определения преобладающих эмоций.
            Выдели ключевые проблемы и потребности, которые отражаются в содержании сообщений.
            Проанализируй географию сообщений и источники информации, указав регионы и каналы распространения данных.
            Используй простой текстовый формат без использования символов форматирования (например, ##, **, ---) и других специальных знаков.
            Если предоставленная информация неполная или требует уточнения, задай дополнительные вопросы для получения более детальной информации, чтобы улучшить качество анализа.

            Для анализа используй первые 250 сообщений:
            {chr(10).join(messages[:250])}
            """

            # Generate report using the selected provider
            system_message = "Ты опытный аналитик данных и социальный исследователь, способный выполнять глубокий контент-анализ с выявлением трендов и инсайтов."
            report_content = self.provider.generate(
                prompt,
                system_message=system_message,
                max_tokens=self.max_tokens
            )
            
            if not report_content:
                return "LLM report unavailable - generation failed"

            # Ensure report directory exists and save with consistent name
            if not self.active_geo:
                raise ValueError("Active geo region is not set")
            
            report_dir = os.path.join("reports", self.active_geo, "llm_reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "detailed_report.txt")
            self.save_report(report_content, report_path)
            print(f"Saved LLM report to: {report_path}")
            print("Детальный отчет GPT успешно создан.")

            # Генерация PDF версии
            from src.reporting.pdf_reporter import PDFReporter
            pdf_reporter = PDFReporter(
                input_data_path=os.path.join(self.input_data_path, self.active_geo, "llm_reports"),
                output_dir=os.path.join(self.output_dir, self.active_geo),
                active_geo=self.active_geo
            )
            pdf_reporter.generate_report(report_content)
            print("PDF версия отчета создана.")

            return report_content

        except Exception as e:
            print(f"Ошибка при генерации отчета через GPT: {e}")
            return f"LLM report unavailable - error: {e}"

    def generate_short_report(self):
        """
        Создает аналитический отчет с ключевыми выводами.
        """
        try:
            # Проверка на наличие файла перед загрузкой
            messages_file_path = os.path.join(self.input_data_path, self.active_geo, "processed_messages.csv")
            if not os.path.exists(messages_file_path):
                print(f"Файл {messages_file_path} не найден.")
                return "No messages data found"

            # Загружаем данные сообщений
            messages_df = pd.read_csv(messages_file_path)
            messages = messages_df["text"].dropna().tolist()

            if not messages:
                print("Нет данных для анализа.")
                return "No messages to analyze"

            # Загружаем данные анализа тональности, если доступны
            sentiment_stats = {}
            sentiment_path = os.path.join(self.input_data_path, self.active_geo, "sentiment_analysis.csv")
            if os.path.exists(sentiment_path):
                sentiment_df = pd.read_csv(sentiment_path)
                sentiment_stats = {
                    'positive': len(sentiment_df[sentiment_df['category'] == 'позитив']),
                    'negative': len(sentiment_df[sentiment_df['category'] == 'негатив']),
                    'neutral': len(sentiment_df[sentiment_df['category'] == 'нейтрально']),
                    'total': len(sentiment_df)
                }

            # Загружаем топ фраз, если доступны
            phrases_stats = []
            phrases_path = os.path.join(self.input_data_path, self.active_geo, "top_phrases.csv")
            if os.path.exists(phrases_path):
                phrases_df = pd.read_csv(phrases_path)
                phrases_stats = phrases_df.to_dict('records')

            # Формирование краткого промпта
            prompt = f"""
            Создай информативный  аналитический отчет  со следующими ключевыми выводами:

            Основные показатели:
            - Общее количество сообщений: {len(messages)}
            {f"- Позитивных сообщений: {sentiment_stats.get('positive', 'N/A')}" if sentiment_stats else ""}
            {f"- Негативных сообщений: {sentiment_stats.get('negative', 'N/A')}" if sentiment_stats else ""}
            {f"- Топ фраз: {chr(10).join([f'{p["phrase"]} ({p["count"]})' for p in phrases_stats[:5]])}" if phrases_stats else ""}

            Требования к анализу:
            Используй простой текстовый формат без использования символов форматирования (например, ##, **, ---) и других специальных знаков.
            Remove markdown formatting (##, **, ---)
            Remove markdown formatting ( **word**)
            1. Выдели  ключевые темы
            2. Укажи общий эмоциональный фон
            3. Отметь  важные выводы
            4. Укажи источники информации
            5. Укажи географию сообщений
            6. Используй простой текстовый формат без markdown разметки
            7. Remove markdown formatting (##, **, ---)
            

            Пример сообщений:
            {chr(10).join(messages[:150])}
            """

            # Generate report using the selected provider
            system_message = "Ты аналитик данных, создающий краткие и информативные отчеты с ключевыми выводами."
            report_content = self.provider.generate(
                prompt,
                system_message=system_message,
                max_tokens=4000  # Shorter output
            )
            
            if not report_content:
                return "Short LLM report unavailable - generation failed"

            # Сохранение отчета
            try:
                self.save_report(report_content, f"short_llm_report_{self.active_geo}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
                print("Краткий отчет успешно создан.")
            except Exception as e:
                print(f"Ошибка сохранения текстового отчета: {e}")
                raise

            # Генерация PDF версии
            try:
                from src.reporting.pdf_reporter import PDFReporter
                pdf_reporter = PDFReporter(
                    input_data_path=os.path.join(self.input_data_path, self.active_geo, "llm_reports"),
                    output_dir=os.path.join(self.output_dir, self.active_geo),
                    active_geo=self.active_geo
                )
                pdf_reporter.generate_short_report(report_content)
                print("PDF версия краткого отчета создана.")
            except Exception as e:
                print(f"Ошибка создания PDF отчета: {e}")
                raise

            return report_content

        except Exception as e:
            error_msg = f"Ошибка при генерации краткого отчета: {str(e)}"
            print(error_msg)
            # Проверяем доступность директории reports
            if not os.path.exists(self.output_dir):
                error_msg += f"\nДиректория {self.output_dir} не существует"
            elif not os.access(self.output_dir, os.W_OK):
                error_msg += f"\nНет прав на запись в директорию {self.output_dir}"
            return error_msg

def main():
    reporter = LLMReporter()
    report = reporter.generate_report()
    print(report)

if __name__ == "__main__":
    main()
