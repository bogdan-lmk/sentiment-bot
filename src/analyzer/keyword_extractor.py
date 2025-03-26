import pandas as pd
import re
import nltk
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.analyzer.santiment import SentimentAnalyzer
import schedule
import time
import logging
import os
from datetime import datetime, timedelta

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class AdvancedNLPAnalyzer:
    def __init__(self, language='russian, ukrainian, english'):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename='nlp_analysis.log'
        )
        self.logger = logging.getLogger(__name__)
        
        # Language and stopwords
        self.language = language
        self.stop_words = self._get_stopwords(language)
        
        # Multilingual themes and need categories
        self.themes = self._get_multilingual_themes()
        self.need_patterns = self._get_multilingual_need_patterns()

    def _get_stopwords(self, language):
        """Get stopwords for multiple languages."""
        languages_list = language.split(', ')
        stop_words = []
        for lang in languages_list:
            try:
                stop_words.extend(stopwords.words(lang))
            except Exception as e:
                self.logger.warning(f"Could not load stopwords for {lang}: {e}")
        return list(set(stop_words))

    def _get_multilingual_themes(self):
        """Define multilingual themes with keywords for Russian, Ukrainian, and English."""
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
        """Define multilingual need patterns."""
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

    def preprocess_text(self, text):
        """Clean and tokenize text for multiple languages."""
        if not isinstance(text, str):
            return []
        
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation
        tokens = word_tokenize(text)
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]

    def sentiment_analysis(self, messages):
        """Perform sentiment analysis on messages using SentimentAnalyzer."""
        analyzer = SentimentAnalyzer(language='russian')
        # Create temporary DataFrame if messages isn't already one
        if not isinstance(messages, pd.DataFrame):
            df = pd.DataFrame({'text': messages})
        else:
            df = messages.copy()
        return analyzer.analyze_messages_from_csv(df)

    def theme_classification(self, messages):
        """Classify messages into predefined themes across multiple languages."""
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

    def identify_needs(self, messages):
        """Identify potential needs and pain points across multiple languages."""
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

    def cluster_messages(self, messages, n_clusters=5):
        """Perform message clustering using TF-IDF and KMeans."""
        if not messages:
            return []
        
        vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        tfidf_matrix = vectorizer.fit_transform(messages)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(tfidf_matrix)
        
        return list(zip(messages, kmeans.labels_))

    def trend_analysis(self, df, time_column='date', window_days=7):
        """Analyze trends in message frequency."""
        df[time_column] = pd.to_datetime(df[time_column])
        trend_data = df.resample(f'{window_days}D', on=time_column).size()
        return trend_data

    def comprehensive_analysis(self, input_path="data/raw/messages.csv", output_dir="data/processed"):
        """Perform comprehensive NLP analysis."""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Read messages
        df = pd.read_csv(input_path)
        messages = df['text'].dropna().tolist()
        
        # Perform analyses
        self.logger.info("Starting comprehensive NLP analysis")
        
        # 1. Keywords
        keywords = Counter(self.preprocess_text(' '.join(messages))).most_common(20)
        pd.DataFrame(keywords, columns=['keyword', 'count']).to_csv(
            f'{output_dir}/keywords.csv', index=False, encoding='utf-8'
        )
        
        # 2. Sentiment Analysis (include date from original data)
        sentiments = self.sentiment_analysis(messages)
        sentiment_df = pd.DataFrame(sentiments)
        if 'date' in df.columns:
            sentiment_df['date'] = df['date'].iloc[:len(sentiment_df)]
        sentiment_df.to_csv(
            f'{output_dir}/sentiment_analysis.csv', index=False, encoding='utf-8'
        )
        
        # Save processed messages with original structure
        df.to_csv(f'{output_dir}/messages.csv', index=False, encoding='utf-8')
        
        # 3. Theme Classification
        themes = self.theme_classification(messages)
        pd.DataFrame(themes).to_csv(
            f'{output_dir}/theme_classification.csv', index=False, encoding='utf-8'
        )
        
        # 4. Needs Identification
        needs = self.identify_needs(messages)
        pd.DataFrame(needs).to_csv(
            f'{output_dir}/needs_analysis.csv', index=False, encoding='utf-8'
        )
        
        # 5. Message Clustering
        clusters = self.cluster_messages(messages)
        pd.DataFrame(clusters, columns=['message', 'cluster']).to_csv(
            f'{output_dir}/message_clusters.csv', index=False, encoding='utf-8'
        )
        
        # 6. Trend Analysis (if date column exists)
        if 'date' in df.columns:
            trends = self.trend_analysis(df)
            trends.to_csv(f'{output_dir}/message_trends.csv')
        
        self.logger.info("Comprehensive NLP analysis completed")

    def schedule_daily_analysis(self, input_path="data/raw/messages.csv", output_dir="data/processed"):
        """Schedule daily NLP analysis."""
        def job():
            try:
                self.comprehensive_analysis(input_path, output_dir)
                self.logger.info("Scheduled daily analysis completed successfully")
            except Exception as e:
                self.logger.error(f"Scheduled analysis failed: {e}")

        # Run immediately and then schedule
        job()
        schedule.every().day.at("00:00").do(job)

        while True:
            schedule.run_pending()
            time.sleep(1)

def main():
    # Создаем анализатор с поддержкой русского, украинского и английского языков
    analyzer = AdvancedNLPAnalyzer(language='russian, ukrainian, english')
    analyzer.comprehensive_analysis()
    # Uncomment the following line to start scheduled daily analysis
    # analyzer.schedule_daily_analysis()

if __name__ == "__main__":
    main()
