import pandas as pd
import os
from src.reporting.base_reporter import BaseReporter

class CSVReporter(BaseReporter):
    """
    Класс для генерации текстового отчета из обработанных CSV данных.
    """
    
    def generate_report(self):
        """
        Генерирует текстовый отчет из CSV-файлов с анализом данных.
        """
        try:
            # Загрузка данных с проверкой на наличие файла
            sentiment_data = self.load_data("sentiment_analysis.csv")
            keywords_data = self.load_data("keywords.csv")
            themes_data = self.load_data("theme_classification.csv")
            needs_data = self.load_data("needs_analysis.csv")
            
            report_content = "Отчет по анализу сообщений\n"
            report_content += "=" * 50 + "\n\n"
            
            # Анализ тональности
            report_content += "Анализ тональности:\n"
            if 'category' in sentiment_data.columns:
                sentiment_summary = sentiment_data["category"].value_counts().to_string()
                report_content += sentiment_summary + "\n\n"
            else:
                report_content += "Не найден столбец 'category' в sentiment_analysis.csv\n\n"
            
            # Топ ключевых слов
            report_content += "Топ ключевых слов:\n"
            if 'keywords' in keywords_data.columns:
                keywords_summary = keywords_data.head(20).to_string(index=False)
                report_content += keywords_summary + "\n\n"
            else:
                report_content += "Не найден столбец 'keywords' в keywords.csv\n\n"
            
            # Классификация тем
            report_content += "Классификация тем:\n"
            if 'themes' in themes_data.columns:
                if themes_data['themes'].apply(lambda x: isinstance(x, list)).all():
                    themes_summary = themes_data.explode("themes")["themes"].value_counts().to_string()
                    report_content += themes_summary + "\n\n"
                else:
                    report_content += "Столбец 'themes' не является списком.\n\n"
            else:
                report_content += "Не найден столбец 'themes' в theme_classification.csv\n\n"
            
            # Выявленные потребности
            report_content += "Выявленные потребности:\n"
            if 'needs' in needs_data.columns:
                if needs_data['needs'].apply(lambda x: isinstance(x, list)).all():
                    needs_summary = needs_data.explode("needs")["needs"].value_counts().to_string()
                    report_content += needs_summary + "\n\n"
                else:
                    report_content += "Столбец 'needs' не является списком.\n\n"
            else:
                report_content += "Не найден столбец 'needs' в needs_analysis.csv\n\n"
            
            self.save_report(report_content, "text_report.txt")
            print("Текстовый отчет успешно создан.")
        except FileNotFoundError as e:
            print(f"Файл не найден: {e}")
        except Exception as e:
            print(f"Ошибка при генерации отчета: {e}")
