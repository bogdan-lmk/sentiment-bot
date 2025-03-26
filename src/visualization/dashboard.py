import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.charts import DataVisualizer  # Assuming charts.py is inside the src/visualization directory

# Load processed data (e.g., sentiment analysis, keywords, etc.)
def load_data():
    sentiment_df = pd.read_csv('data/processed/sentiment_analysis.csv')
    try:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    except Exception as e:
        st.warning(f"Could not parse date column: {e}")
    keywords_df = pd.read_csv('data/processed/keywords.csv')
    trend_df = pd.read_csv('data/processed/message_trends.csv')
    trend_df['date'] = pd.to_datetime(trend_df['date'])
    return sentiment_df, keywords_df, trend_df

# Create the Streamlit Dashboard
def create_dashboard():
    # Page Title
    st.title('Telegram Chat Analysis Dashboard')

    # Load data
    sentiment_df, keywords_df, trend_df = load_data()

    # Display sentiment analysis summary
    st.subheader('Sentiment Analysis')
    sentiment_count = sentiment_df['category'].value_counts()
    st.write('### Sentiment Distribution')
    st.bar_chart(sentiment_count)

    # Display top keywords
    st.subheader('Top Keywords')
    top_keywords = keywords_df.head(10)
    st.write(top_keywords[['keyword', 'count']])

    # Display trend analysis
    st.subheader('Message Trends')
    st.write('### Message Frequency Over Time')
    trend_df['date'] = pd.to_datetime(trend_df['date'])
    trend_data = trend_df.set_index('date').resample('D').size()
    st.line_chart(trend_data)

    # Add a section to show different charts based on user choice
    chart_option = st.selectbox('Select a Chart to View:', ['Sentiment Distribution', 'Top Keywords', 'Message Trends'])

    if chart_option == 'Sentiment Distribution':
        st.write('### Sentiment Distribution by Time of Day')
        if 'date' in sentiment_df.columns:
            visualizer = DataVisualizer(sentiment_df)
            fig = visualizer.plot_sentiment_by_time(sentiment_df)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Date column not found in sentiment data - cannot show time distribution")

    elif chart_option == 'Top Keywords':
        st.write('### Keyword Frequency')
        visualizer = DataVisualizer(keywords_df)
        fig = visualizer.plot_top_keywords(keywords_df)
        st.pyplot(fig)
        plt.close(fig)

    elif chart_option == 'Message Trends':
        st.write('### Trends in Messages')
        visualizer = DataVisualizer(trend_df)
        fig = visualizer.plot_trends(trend_data)
        st.pyplot(fig)
        plt.close(fig)

    # Optional: Downloadable CSV for sentiment analysis
    st.subheader('Downloadable Reports')
    csv = sentiment_df.to_csv(index=False)
    st.download_button(
        label="Download Sentiment Analysis CSV",
        data=csv,
        file_name="sentiment_analysis.csv",
        mime="text/csv"
    )

    # Optional: Show the raw message data (if necessary)
    if st.checkbox('Show Raw Messages Data'):
        raw_data = pd.read_csv('data/raw/messages.csv')
        st.write(raw_data.head(20))

# Run the dashboard
if __name__ == '__main__':
    create_dashboard()
