from fpdf import FPDF
import os
import requests
from pathlib import Path
from src.reporting.base_reporter import BaseReporter

# Path to Unicode font that supports Cyrillic
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fonts")
FONT_PATH = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_URL = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"

class PDFReporter(BaseReporter):
    """
    Класс для генерации PDF-отчета с улучшенным форматированием и поддержкой Кириллицы.
    """
    
    def __init__(self, input_data_path="data/processed", output_dir="reports"):
        super().__init__()
        self.input_data_path = input_data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _download_font(self):
        """Загрузка и установка шрифта DejaVu с поддержкой Юникода."""
        try:
            if os.path.exists(FONT_PATH):
                return True
                
            os.makedirs(FONT_DIR, exist_ok=True)
            
            print("Загрузка шрифтов DejaVu...")
            response = requests.get(FONT_URL)
            response.raise_for_status()
            
            zip_path = os.path.join(FONT_DIR, "dejavu-fonts.zip")
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            import zipfile
            with zipfile.ZipFile(zip_path) as z:
                for file in z.namelist():
                    if file.endswith("DejaVuSans.ttf"):
                        z.extract(file, FONT_DIR)
                        os.rename(
                            os.path.join(FONT_DIR, file),
                            FONT_PATH
                        )
            
            os.remove(zip_path)
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки шрифтов: {e}")
            return False

    def generate_pdf_report(self, report_text, filename="report.pdf", title="Отчет"):
        """
        Создает PDF-отчет с улучшенным форматированием.
        
        Args:
            report_text (str): Текст отчета
            filename (str): Имя файла PDF
            title (str): Заголовок отчета
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()
        
        # Настройка шрифта с поддержкой Кириллицы
        try:
            if not os.path.exists(FONT_PATH):
                self._download_font()
            
            # Register regular and bold variants
            pdf.add_font('DejaVu', '', FONT_PATH, uni=True)
            pdf.add_font('DejaVu', 'B', FONT_PATH, uni=True)
            pdf.set_font('DejaVu', size=12)
        except Exception as e:
            print(f"Не удалось загрузить специализированный шрифт: {e}")
            pdf.set_font('Arial', size=12)
        
        # Установка параметров страницы
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)
        effective_width = pdf.w - 2 * pdf.l_margin
        
        # Добавление заголовка
        pdf.set_font('DejaVu', 'B', 16)
        pdf.cell(effective_width, 15, title, ln=True, align='C')
        pdf.ln(10)
        
        # Установка основного шрифта
        pdf.set_font('DejaVu', size=12)
        line_height = 10
        
        # Обработка текста с сохранением структуры
        try:
            # Ensures clean UTF-8 encoding
            report_text = report_text.encode('utf-8', errors='replace').decode('utf-8')
            
            # Разбор текста на абзацы и структурные элементы
            for paragraph in report_text.split('\n'):
                if paragraph.strip():
                    # Добавление отступа для заголовков
                    if paragraph.startswith(('###', '##', '#')):
                        pdf.set_font('DejaVu', 'B', 14)
                        paragraph = paragraph.lstrip('#').strip()
                        pdf.cell(effective_width, line_height, paragraph, ln=True)
                        pdf.ln(5)
                        pdf.set_font('DejaVu', size=12)
                    else:
                        pdf.multi_cell(effective_width, line_height, paragraph, align='L')
                
                pdf.ln(5)  # Межабзацный интервал
        
        except Exception as e:
            print(f"Ошибка при обработке текста: {e}")
            pdf.multi_cell(effective_width, line_height, str(report_text), align='L')
        
        # Сохранение PDF
        output_path = os.path.join(self.output_dir, filename)
        pdf.output(output_path)
        print(f"PDF-отчет сохранен в {output_path}")
        return output_path
    
    def load_data(self, filename):
        """
        Загружает текстовый файл из reports или data/processed.
        """
        # First check reports directory
        reports_path = os.path.join(self.output_dir, filename)
        if os.path.exists(reports_path):
            with open(reports_path, "r", encoding="utf-8") as f:
                return f.read()
        
        # Fall back to input data path
        data_path = os.path.join(self.input_data_path, filename)
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                return f.read()
        
        raise FileNotFoundError(f"Файл {filename} не найден ни в {self.output_dir}, ни в {self.input_data_path}")

    def generate_report(self, report_text=None, chat_names=None):
        """
        Создание PDF-отчета с поддержкой списка чатов.
        """
        try:
            if report_text is None:
                report_text = self.load_data("detailed_llm_report.txt")
                if not report_text:
                    print("Отчет не найден.")
                    return
            
            if report_text:
                title = "Анализ Сообщений"
                if chat_names:
                    title += f" ({', '.join(chat_names)})"
                
                self.generate_pdf_report(report_text, "llm_report.pdf", title)
        except Exception as e:
            print(f"Ошибка при создании PDF-отчета: {e}")

    def generate_short_report(self, report_text=None, chat_names=None):
        """
        Создание краткого PDF-отчета.
        """
        try:
            if report_text is None:
                report_text = self.load_data("short_llm_report.txt")
                if not report_text:
                    print("Краткий отчет не найден.")
                    return
            
            if report_text:
                title = "Краткий Анализ Сообщений"
                if chat_names:
                    title += f" ({', '.join(chat_names)})"
                
                self.generate_pdf_report(report_text, "llm_short_report.pdf", title)
        except Exception as e:
            print(f"Ошибка при создании краткого PDF-отчета: {e}")
