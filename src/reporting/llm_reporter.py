import os
import pandas as pd
from collections import Counter
from datetime import datetime
import logging
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
        # Validate active_geo
        if active_geo == "--geo":
            raise ValueError("Invalid geo code: '--geo' is not allowed")
            
        # Initialize with base paths
        super().__init__(input_data_path=input_data_path, output_dir=output_dir)
        
        # Set base output directory
        self.output_dir = os.path.abspath(output_dir)
        
        # Set instance variables
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
            # Validate active_geo is set and valid
            if not self.active_geo:
                error_msg = "Active geo region is not set"
                print(error_msg)
                return error_msg
            
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
                # Convert sentiment scores to categories
                sentiment_df['category'] = sentiment_df['sentiment'].apply(
                    lambda x: 'позитив' if x in ['POSITIVE', 'positive', 4, 5] 
                    else 'негатив' if x in ['NEGATIVE', 'negative', 1, 2] 
                    else 'нейтрально'
                )
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

            # Create report directory structure
            report_dir = os.path.join(self.output_dir, self.active_geo, "llm")
            os.makedirs(report_dir, exist_ok=True)
            
            # Create report file
            report_filename = "report.txt"
            report_path = os.path.join(report_dir, report_filename)
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"Saved LLM report to: {report_path}")

            # Generate PDF version
            try:
                from src.reporting.pdf_reporter import PDFReporter
                pdf_dir = os.path.join(self.output_dir, self.active_geo, "pdf")
                os.makedirs(pdf_dir, exist_ok=True)
                
                pdf_reporter = PDFReporter(
                    input_data_path=report_dir,
                    output_dir=pdf_dir,
                    active_geo=self.active_geo
                )
                pdf_reporter.generate_report(report_content)
                print(f"PDF report saved to: {os.path.join(pdf_dir, 'report.pdf')}")
            except Exception as e:
                print(f"Ошибка при создании PDF: {e}")
                # Continue even if PDF creation fails

            return report_content

        except Exception as e:
            print(f"Ошибка при генерации отчета через GPT: {e}")
            return f"LLM report unavailable - error: {e}"

    def generate_short_report(self):
        """
        Создает аналитический отчет с ключевыми выводами.
        """
        try:
            # Validate active_geo is set and valid
            if not self.active_geo:
                error_msg = "Active geo region is not set"
                print(error_msg)
                return error_msg
            if self.active_geo == "--geo":
                raise ValueError("Invalid geo code: '--geo' is not allowed")

            # Check for required data files
            messages_file_path = os.path.join(self.input_data_path, self.active_geo, "processed_messages.csv")
            if not os.path.exists(messages_file_path):
                error_msg = f"Required data file not found: {messages_file_path}"
                print(error_msg)
                return error_msg

            # Load message data
            messages_df = pd.read_csv(messages_file_path)
            messages = messages_df["text"].dropna().tolist()

            if not messages:
                print("Нет данных для анализа.")
                return "No messages to analyze"

            # Load sentiment data if available
            sentiment_stats = {}
            sentiment_path = os.path.join(self.input_data_path, self.active_geo, "sentiment_analysis.csv")
            if os.path.exists(sentiment_path):
                sentiment_df = pd.read_csv(sentiment_path)
                sentiment_df['category'] = sentiment_df['sentiment'].apply(
                    lambda x: 'позитив' if x in ['POSITIVE', 'positive', 4, 5] 
                    else 'негатив' if x in ['NEGATIVE', 'negative', 1, 2] 
                    else 'нейтрально'
                )
                sentiment_stats = {
                    'positive': len(sentiment_df[sentiment_df['category'] == 'позитив']),
                    'negative': len(sentiment_df[sentiment_df['category'] == 'негатив']),
                    'neutral': len(sentiment_df[sentiment_df['category'] == 'нейтрально']),
                    'total': len(sentiment_df)
                }

            # Load top phrases if available
            phrases_stats = []
            phrases_path = os.path.join(self.input_data_path, self.active_geo, "top_phrases.csv")
            if os.path.exists(phrases_path):
                phrases_df = pd.read_csv(phrases_path)
                phrases_stats = phrases_df.to_dict('records')

            # Create short report prompt
            prompt = f"""
            Создай информативный аналитический отчет со следующими ключевыми выводами:

            Основные показатели:
            - Общее количество сообщений: {len(messages)}
            {f"- Позитивных сообщений: {sentiment_stats.get('positive', 'N/A')}" if sentiment_stats else ""}
            {f"- Негативных сообщений: {sentiment_stats.get('negative', 'N/A')}" if sentiment_stats else ""}
            {f"- Топ фраз: {chr(10).join([f'{p['phrase']} ({p['count']})' for p in phrases_stats[:5]])}" if phrases_stats else ""}

            Требования к анализу:
            Используй простой текстовый формат без использования символов форматирования (например, ##, **, ---) и других специальных знаков.
            Remove markdown formatting (##, **, ---)
            Remove markdown formatting ( **word**)
            1. Выдели ключевые темы
            2. Укажи общий эмоциональный фон
            3. Отметь важные выводы
            4. Укажи источники информации
            5. Укажи географию сообщений
            6. Используй простой текстовый формат без markdown разметки
            7. Remove markdown formatting (##, **, ---)
            

            Пример сообщений:
            {chr(10).join(messages[:150])}
            """

            # Generate short report
            system_message = "Ты аналитик данных, создающий краткие и информативные отчеты с ключевыми выводами."
            report_content = self.provider.generate(
                prompt,
                system_message=system_message,
                max_tokens=4000  # Shorter output
            )
            
            if not report_content:
                return "Short LLM report unavailable - generation failed"

            # Create short report directory
            short_dir = os.path.join(self.output_dir, self.active_geo, "llm", "short")
            os.makedirs(short_dir, exist_ok=True)
            
            # Create short report file
            short_report_path = os.path.join(short_dir, "short_report.txt")
            with open(short_report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"Saved short report to: {short_report_path}")

            # Generate PDF version
            try:
                from src.reporting.pdf_reporter import PDFReporter
                pdf_short_dir = os.path.join(self.output_dir, self.active_geo, "pdf", "short")
                os.makedirs(pdf_short_dir, exist_ok=True)
                
                pdf_reporter = PDFReporter(
                    input_data_path=short_dir,
                    output_dir=pdf_short_dir,
                    active_geo=self.active_geo
                )
                pdf_reporter.generate_short_report(report_content)
                print(f"PDF short report saved to: {os.path.join(pdf_short_dir, 'short_report.pdf')}")
            except Exception as e:
                print(f"Ошибка создания PDF отчета: {str(e)}")
                # Continue even if PDF creation fails

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

    def save_report(self, content, file_path):
        """
        Safely saves report content to the specified file_path.
        This method is kept but we're using direct file writing in the main methods for clarity.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write content to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error saving report: {e}")
            return False

def main():
    # Example usage with required geo parameter
    reporter = LLMReporter()  # Use a valid geo code
    print("Generating detailed report...")
    detailed_report = reporter.generate_report()
    if detailed_report and detailed_report != "No messages data found":
        print("Detailed report generated successfully")
    
    print("\nGenerating short report...")
    short_report = reporter.generate_short_report()
    if short_report and short_report != "No messages data found":
        print("Short report generated successfully")

if __name__ == "__main__":
    main()
