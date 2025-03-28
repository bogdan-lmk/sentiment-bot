import sys
import os
import logging
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Append project root to system path
sys.path.append('/Users/buyer7/Desktop/ai-santiment')
from src.visualization.charts import DataVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Загрузка и предобработка данных с расширенной обработкой ошибок.
    
    :return: Кортеж с DataFrame sentiment, keywords и trends
    """
    try:
        # Загрузка данных с проверкой существования файлов
        data_files = {
            'sentiment': 'data/processed/sentiment_analysis.csv',
            'keywords': 'data/processed/keywords.csv',
            'trends': 'data/processed/message_trends.csv'
        }
        
        # Проверка существования файлов
        for name, path in data_files.items():
            if not os.path.exists(path):
                st.error(f"Файл {name} не найден: {path}")
                return None, None, None
        
        # Загрузка DataFrame
        sentiment_df = pd.read_csv(data_files['sentiment'])
        keywords_df = pd.read_csv(data_files['keywords'])
        trend_df = pd.read_csv(data_files['trends'])
        
        # Обработка даты с расширенной обработкой ошибок
        for df, col_name in [(sentiment_df, 'date'), (trend_df, 'date')]:
            try:
                df[col_name] = pd.to_datetime(df[col_name])
            except Exception as e:
                st.warning(f"Ошибка парсинга даты в {col_name}: {e}")
        
        return sentiment_df, keywords_df, trend_df
    
    except Exception as e:
        st.error(f"Критическая ошибка загрузки данных: {e}")
        logger.error(f"Ошибка загрузки данных: {e}")
        return None, None, None

def create_advanced_sentiment_chart(sentiment_df):
    """
    Создание расширенной визуализации тональности.
    
    :param sentiment_df: DataFrame с данными тональности
    """
    try:
        plt.figure(figsize=(12, 6))
        
        # Определение колонки тональности
        sentiment_col = 'sentiment' if 'sentiment' in sentiment_df.columns else 'category'
        
        # Расширенная визуализация
        sentiment_count = sentiment_df[sentiment_col].value_counts()
        
        plt.subplot(121)
        sns.barplot(x=sentiment_count.index, y=sentiment_count.values, 
                   hue=sentiment_count.index, palette='viridis', legend=False)
        plt.title('Распределение тональности', fontsize=12)
        plt.xlabel('Тональность', fontsize=10)
        plt.ylabel('Количество', fontsize=10)
        plt.xticks(rotation=45)
        
        plt.subplot(122)
        plt.pie(sentiment_count.values, labels=sentiment_count.index, autopct='%1.1f%%', 
                colors=sns.color_palette('viridis'))
        plt.title('Доля тональностей', fontsize=12)
        
        plt.tight_layout()
        return plt.gcf()
    
    except Exception as e:
        st.error(f"Ошибка создания графика тональности: {e}")
        logger.error(f"Ошибка создания графика тональности: {e}")
        return None

def create_dashboard():
    """
    Создание интерактивной панели управления в Streamlit.
    """
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    # Настройка страницы
    st.set_page_config(
        page_title="Telegram Chat Analysis", 
        page_icon=":chart_with_upwards_trend:", 
        layout="wide"
    )
    
    # Заголовок
    st.title('🤖 Telegram Chat Analysis Dashboard')
    
    # Загрузка данных
    sentiment_df, keywords_df, trend_df = load_data()
    
    if sentiment_df is None or keywords_df is None or trend_df is None:
        st.error("Не удалось загрузить данные. Проверьте источники данных.")
        return
    
    # Создание колонок для компоновки
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('📊 Sentiment Distribution')
        # Расширенный график тональности
        sentiment_fig = create_advanced_sentiment_chart(sentiment_df)
        if sentiment_fig:
            st.pyplot(sentiment_fig)
            plt.close(sentiment_fig)
    
    with col2:
        st.subheader('🔑 Top Keywords')
        # Интерактивная таблица ключевых слов
        top_keywords = keywords_df.head(10)
        st.dataframe(
            top_keywords[['keyword', 'count']].style.background_gradient(cmap='Blues'),
            use_container_width=True
        )
    
    # Секция трендов
    st.subheader('📈 Message Trends')
    
    # Интерактивный выбор временного масштаба
    time_scale = st.radio(
        "Выберите временной масштаб:", 
        ['Ежедневно', 'Еженедельно', 'Ежемесячно']
    )
    
    # Динамическая агрегация тренда
    time_map = {
        'Ежедневно': 'D',
        'Еженедельно': 'W',
        'Ежемесячно': 'M'
    }
    
    trend_data = trend_df.set_index('date').resample(time_map[time_scale]).size()
    
    # Улучшенный график трендов
    plt.figure(figsize=(12, 4))
    trend_data.plot(kind='line', marker='o', linestyle='-', color='green')
    plt.title(f'Тренды сообщений ({time_scale})', fontsize=12)
    plt.xlabel('Дата', fontsize=10)
    plt.ylabel('Количество сообщений', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
    
    # Дополнительные опции
    st.sidebar.header('Дополнительные инструменты')
    
    # Загрузка CSV
    with open('data/processed/sentiment_analysis.csv', 'rb') as file:
        st.sidebar.download_button(
            label="📥 Скачать Sentiment CSV",
            data=file,
            file_name="sentiment_analysis.csv",
            mime="text/csv"
        )
    
    # Просмотр сырых данных
    if st.sidebar.checkbox('🔍 Показать сырые сообщения'):
        try:
            raw_data = pd.read_csv('data/raw/messages.csv')
            st.subheader('Первые 20 сырых сообщений')
            st.dataframe(raw_data.head(20))
        except Exception as e:
            st.error(f"Ошибка загрузки сырых данных: {e}")

if __name__ == '__main__':
    try:
        import asyncio
        # Ensure we have a running event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        create_dashboard()
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        st.error(f"Critical dashboard error: {e}")
