import pandas as pd
import re
import numpy as np
from typing import List, Dict, Any
import logging
import os
import schedule
import time
from datetime import datetime, timedelta
from typing import List, Tuple
# Замена устаревших библиотек
import spacy
from spacy.lang.ru import Russian
from spacy.lang.uk import Ukrainian
from spacy.lang.en import English
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

class AdvancedNLPAnalyzer:
    def __init__(self, language='russian, ukrainian, english'):
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename='nlp_analysis.log'
        )
        self.logger = logging.getLogger(__name__)
        
        # Инициализация языковых моделей
        self.nlp_models = {
            'russian': spacy.load('ru_core_news_sm'),
            'ukrainian': spacy.load('uk_core_news_sm'),
            'english': spacy.load('en_core_web_sm')
        }
        
        # Многоязычный sentiment анализ
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        # Язык и стоп-слова
        self.language = language
        self.stop_words = self._get_stopwords(language)
        
        # Мультиязычные темы и паттерны потребностей
        self.themes = self._get_multilingual_themes()
        self.need_patterns = self._get_multilingual_need_patterns()

    def _get_stopwords(self, language):
        """Получение стоп-слов для нескольких языков."""
        stop_words = set()
        for lang in language.split(', '):
            if lang == 'russian':
                stop_words.update(Russian().Defaults.stop_words)
            elif lang == 'ukrainian':
                stop_words.update(Ukrainian().Defaults.stop_words)
            elif lang == 'english':
                stop_words.update(English().Defaults.stop_words)
        return list(stop_words)

    def _get_multilingual_themes(self):
        """Определение мультиязычных тем."""
        return {
            'housing': {
                'russian': ['квартира', 'дом', 'комната', 'аренда', 'съём'],
                'ukrainian': ['квартира', 'будинок', 'кімната', 'оренда', 'проживання'],
                'english': ['apartment', 'house', 'room', 'rent', 'accommodation']
            },
            'work': {
                'russian': ['вакансия', 'работа', 'зарплата', 'профессия', 'карьера'],
                'ukrainian': ['вакансія', 'робота', 'зарплата', 'професія', 'кар\'єра'],
                'english': ['vacancy', 'job', 'salary', 'profession', 'career']
            },
            'adaptation': {
                'russian': ['страна', 'город', 'привыкание', 'культура', 'язык'],
                'ukrainian': ['країна', 'місто', 'адаптація', 'культура', 'мова'],
                'english': ['country', 'city', 'adaptation', 'culture', 'language']
            },
            'documents': {
                'russian': ['виза', 'паспорт', 'регистрация', 'разрешение', 'документ'],
                'ukrainian': ['віза', 'паспорт', 'реєстрація', 'дозвіл', 'документ'],
                'english': ['visa', 'passport', 'registration', 'permit', 'document']
            },
            'finances': {
                'russian': ['налог', 'банк', 'счёт', 'деньги', 'страховка'],
                'ukrainian': ['податок', 'банк', 'рахунок', 'гроші', 'страхування'],
                'english': ['tax', 'bank', 'account', 'money', 'insurance']
            }
        }

    def _get_multilingual_need_patterns(self):
        """Определение мультиязычных паттернов потребностей."""
        return {
            'no_housing': {
                'russian': ['нет', 'жильё', 'комната', 'квартира', 'проживание'],
                'ukrainian': ['немає', 'житло', 'кімната', 'квартира', 'проживання'],
                'english': ['no', 'housing', 'room', 'apartment', 'accommodation']
            },
            'document_issues': {
                'russian': ['проблема', 'документ', 'виза', 'паспорт'],
                'ukrainian': ['проблема', 'документ', 'віза', 'паспорт'],
                'english': ['problem', 'document', 'visa', 'passport']
            },
            'job_search': {
                'russian': ['ищу', 'работа', 'вакансия', 'трудоустройство'],
                'ukrainian': ['шукаю', 'робота', 'вакансія', 'працевлаштування'],
                'english': ['looking', 'job', 'vacancy', 'employment']
            },
            'need_lawyer': {
                'russian': ['юрист', 'адвокат', 'консультация', 'право'],
                'ukrainian': ['юрист', 'адвокат', 'консультація', 'право'],
                'english': ['lawyer', 'attorney', 'consultation', 'legal']
            },
            'tax_issues': {
                'russian': ['налог', 'проблема', 'непонятно', 'сложно'],
                'ukrainian': ['податок', 'проблема', 'незрозуміло', 'складно'],
                'english': ['tax', 'problem', 'unclear', 'complicated']
            }
        }

    def preprocess_text(self, text: str) -> List[str]:
        """Очистка и токенизация текста для нескольких языков."""
        if not isinstance(text, str):
            return []
        
        text = re.sub(r'http\S+', '', text)  # Удаление URL
        text = re.sub(r'[^\w\s]', '', text.lower())  # Удаление пунктуации
        
        # Выбор модели в зависимости от языка
        nlp_model = self.nlp_models.get('russian', self.nlp_models['english'])
        doc = nlp_model(text)
        
        return [token.text for token in doc if 
                token.text not in self.stop_words and len(token.text) > 2]

    def sentiment_analysis(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Многоязычный анализ тональности."""
        sentiments = []
        for message in messages:
            try:
                # Truncate messages longer than 500 characters to avoid model errors
                truncated_msg = message[:500] if len(message) > 500 else message
                result = self.sentiment_analyzer(truncated_msg)[0]
                sentiments.append({
                    'text': message,
                    'sentiment': result['label'],
                    'score': result['score']
                })
            except Exception as e:
                self.logger.warning(f"Failed to analyze sentiment for message: {e}")
                sentiments.append({
                    'text': message,
                    'sentiment': 'neutral',
                    'score': 0.5
                })
        return sentiments

    def theme_classification(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Классификация сообщений по темам."""
        classifications = []
        for message in messages:
            if not isinstance(message, str):
                continue
            message_lower = message.lower()
            detected_themes = []
            
            for theme, language_keywords in self.themes.items():
                for lang_keywords in language_keywords.values():
                    if any(keyword in message_lower for keyword in lang_keywords):
                        detected_themes.append(theme)
                        break
            
            classifications.append({
                'text': message,
                'themes': detected_themes or ['general']
            })
        return classifications

    def identify_needs(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Выявление потребностей и болевых точек."""
        needs_analysis = []
        for message in messages:
            if not isinstance(message, str):
                continue
            message_lower = message.lower()
            detected_needs = []
            
            for need, language_patterns in self.need_patterns.items():
                for lang_patterns in language_patterns.values():
                    if any(pattern in message_lower for pattern in lang_patterns):
                        detected_needs.append(need)
                        break
            
            needs_analysis.append({
                'text': message,
                'needs': detected_needs
            })
        return needs_analysis

    def cluster_messages(self, messages: List[str], n_clusters: int = 5) -> List[Tuple[str, int]]:
        """Кластеризация сообщений с использованием TF-IDF и KMeans."""
        if not messages:
            return []
        
        vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        tfidf_matrix = vectorizer.fit_transform(messages)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(tfidf_matrix)
        
        return list(zip(messages, kmeans.labels_))

    def trend_analysis(self, df: pd.DataFrame, time_column: str = 'date', window_days: int = 7) -> pd.Series:
        """Анализ трендов частоты сообщений."""
        df[time_column] = pd.to_datetime(df[time_column])
        trend_data = df.resample(f'{window_days}D', on=time_column).size()
        return trend_data

    def keyword_extractor(self, texts: List[str], top_n: int = 10, min_count: int = 2) -> List[Tuple[str, int]]:
        """Извлекает наиболее часто упоминаемые фразы из текстов.
        
        Args:
            texts: Список текстов для анализа
            top_n: Количество возвращаемых топ фраз
            min_count: Минимальное количество упоминаний фразы
            
        Returns:
            Список кортежей (фраза, количество) отсортированный по частоте
        """
        # Фразы для исключения
        EXCLUDED_PHRASES = {
            'доброго дня', 'добрый день', 'доброго вечора', 
            'підкажіть ласка', 'можливо хтось', 'хтось знає',
            'добрий день', 'здравствуйте подскажите', 'добрый вечер',
            'може хтось', 'дуже вдячний', 'размещения рекламы',
            'рекламы пишите', 'размещения рекламы пишите'
        }

        # Собираем все тексты в один
        combined_text = ' '.join(text for text in texts if isinstance(text, str))
        
        # Обрабатываем текст
        processed_tokens = self.preprocess_text(combined_text)
        
        # Генерируем n-граммы (2-4 слова)
        ngrams = []
        for n in range(2, 5):
            ngrams.extend([' '.join(processed_tokens[i:i+n]) 
                         for i in range(len(processed_tokens)-n+1)])
        
        # Считаем частоту фраз, исключая нежелательные
        phrase_counts = {}
        for phrase in ngrams:
            if phrase.lower() not in EXCLUDED_PHRASES:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Фильтруем по минимальному количеству и сортируем
        filtered_phrases = [(p, c) for p, c in phrase_counts.items() if c >= min_count]
        sorted_phrases = sorted(filtered_phrases, key=lambda x: x[1], reverse=True)
        
        return sorted_phrases[:top_n]

    def generate_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Создание визуализаций."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Распределение тем
        theme_dist = df['theme'].value_counts()
        fig_themes = px.pie(
            names=theme_dist.index, 
            values=theme_dist.values, 
            title='Распределение тем'
        )
        fig_themes.write_html(f'{output_dir}/theme_distribution.html')

        # 2. Тренды тональности
        if 'sentiment' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            sentiment_trends = df.groupby([
                pd.Grouper(key='date', freq='W'), 'theme'
            ])['sentiment'].mean().reset_index()
            
            fig_sentiment = px.line(
                sentiment_trends, 
                x='date', 
                y='sentiment', 
                color='theme',
                title='Динамика тональности по темам'
            )
            fig_sentiment.write_html(f'{output_dir}/sentiment_trends.html')

    def comprehensive_analysis(
        self,
        geo_code: str,
        input_path: str = None,
        output_dir: str = None,
        input_dir: str = None  # Add this parameter to match main.py's call
    ):
        """Полный NLP-анализ для конкретного гео-региона."""
        input_path = input_path or f"data/raw/{geo_code}/messages_{geo_code}.csv"
        output_dir = output_dir or f"data/processed/{geo_code}"
        """Полный NLP-анализ."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Чтение сообщений
        df = pd.read_csv(input_path)
        messages = df['text'].dropna().tolist()
        
        self.logger.info("Начало комплексного NLP-анализа")
        
        # 1. Ключевые слова и фразы
        # Одиночные ключевые слова
        keywords = pd.DataFrame(
            self.preprocess_text(' '.join(messages)), 
            columns=['keyword']
        ).value_counts().reset_index(name='count')
        keywords.to_csv(f'{output_dir}/keywords.csv', index=False, encoding='utf-8')
        
        # Частые фразы (2-4 слова)
        phrases = self.keyword_extractor(messages, top_n=50, min_count=3)
        if phrases:  # Only create file if we have phrases
            phrases_df = pd.DataFrame(phrases, columns=['phrase', 'count'])
            phrases_df.to_csv(
                f'{output_dir}/top_phrases.csv', 
                index=False, 
                encoding='utf-8'
            )
            print(f"Saved top phrases to {output_dir}/top_phrases.csv")
        else:
            print("No phrases found to save")
        
        # 2. Анализ тональности (всегда выполняем)
        sentiments = self.sentiment_analysis(messages)
        sentiment_df = pd.DataFrame(sentiments)
        if 'date' in df.columns:
            sentiment_df['date'] = df['date'].iloc[:len(sentiment_df)]
        sentiment_df.to_csv(f'{output_dir}/sentiment_analysis.csv', index=False, encoding='utf-8')
        
        # Сохраняем обработанные сообщения
        processed_df = df[['date', 'text']].copy()
        processed_df['processed_text'] = [' '.join(self.preprocess_text(t)) for t in processed_df['text']]
        processed_df.to_csv(f'{output_dir}/processed_messages.csv', index=False, encoding='utf-8')
        
        # 3. Классификация тем
        themes = self.theme_classification(messages)
        themes_df = pd.DataFrame(themes)
        themes_df.to_csv(f'{output_dir}/theme_classification.csv', index=False, encoding='utf-8')
        
        # Merge themes back into main DataFrame for visualization
        if len(themes_df) == len(df):
            df['theme'] = themes_df['themes'].apply(lambda x: x[0] if x else 'general')
        
        # 4. Анализ потребностей
        needs = self.identify_needs(messages)
        pd.DataFrame(needs).to_csv(f'{output_dir}/needs_analysis.csv', index=False, encoding='utf-8')
        
        # 5. Кластеризация
        clusters = self.cluster_messages(messages)
        pd.DataFrame(clusters, columns=['message', 'cluster']).to_csv(f'{output_dir}/message_clusters.csv', index=False, encoding='utf-8')
        
        # 6. Анализ трендов
        if 'date' in df.columns:
            trends = self.trend_analysis(df)
            trends.to_csv(f'{output_dir}/message_trends.csv')
        
        # 7. Визуализации
        if 'theme' in df.columns:
            self.generate_visualizations(df, output_dir)
            
        # Генерация визуализаций для фраз
        try:
            if os.path.exists(f'{output_dir}/top_phrases.csv'):
                phrases_df = pd.read_csv(f'{output_dir}/top_phrases.csv')
                if not phrases_df.empty:
                    # Create HTML visualization
                    fig = px.bar(
                        phrases_df.head(20),
                        x='count',
                        y='phrase',
                        orientation='h',
                        title='Top 20 Most Frequent Phrases'
                    )
                    fig.write_html(f'{output_dir}/top_phrases.html')
                    
                    # Create PNG image
                    fig = px.bar(
                        phrases_df.head(20),
                        x='count',
                        y='phrase',
                        orientation='h',
                        title='Top 20 Most Frequent Phrases'
                    )
                    fig.write_image(f'{output_dir}/top_phrases.png')
        except Exception as e:
            self.logger.error(f"Error generating phrase visualizations: {e}")
        
        self.logger.info("Комплексный NLP-анализ завершен")

    def schedule_daily_analysis(
        self, 
        input_path: str = "data/raw/messages.csv", 
        output_dir: str = "data/processed"
    ):
        """Планирование ежедневного NLP-анализа."""
        def job():
            try:
                self.comprehensive_analysis(input_path, output_dir)
                self.logger.info("Ежедневный анализ успешно завершен")
            except Exception as e:
                self.logger.error(f"Ошибка при выполнении анализа: {e}")

        # Выполнение немедленно и затем планирование
        job()
        schedule.every().day.at("00:00").do(job)

        while True:
            schedule.run_pending()
            time.sleep(1)

def main():
    # Создание анализатора с поддержкой русского, украинского и английского языков
    analyzer = AdvancedNLPAnalyzer(language='russian, ukrainian, english')
    analyzer.comprehensive_analysis(geo_code='DEU')
    # Раскомментируйте следующую строку для запуска ежедневного анализа
    # analyzer.schedule_daily_analysis()

if __name__ == "__main__":
    main()
