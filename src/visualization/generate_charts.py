import pandas as pd
import os
from src.visualization.charts import DataVisualizer
import matplotlib.pyplot as plt

def load_data(data_dir):
    """Загружает все CSV файлы из директории"""
    data = {}
    if not os.path.exists(data_dir):
        return data
        
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            try:
                name = file.split('.')[0]
                filepath = os.path.join(data_dir, file)
                if os.path.getsize(filepath) > 0:  # Проверяем что файл не пустой
                    data[name] = pd.read_csv(filepath)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    return data

def save_plot(fig, filename, output_dir='reports/visualizations'):
    """Сохраняет график в файл"""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def main():
    data_dir = 'data/processed'
    data = load_data(data_dir)
    
    visualizer = DataVisualizer(data.get('sentiment_analysis', pd.DataFrame()))
    
    # График распределения тональности
    if 'sentiment_analysis' in data:
        fig = visualizer.plot_sentiment_distribution()
        save_plot(fig, 'sentiment_distribution.png')
    
    # График ключевых слов
    if 'keywords' in data:
        fig = visualizer.plot_top_keywords(data['keywords'])
        save_plot(fig, 'top_keywords.png')
    
    # График трендов сообщений
    if 'message_trends' in data:
        fig = visualizer.plot_trends(data['message_trends'])
        save_plot(fig, 'message_trends.png')
    
    # График тематических кластеров
    if 'theme_classification' in data:
        try:
            if all(col in data['theme_classification'].columns for col in ['x', 'y', 'cluster']):
                fig = visualizer.plot_clusters(data['theme_classification'])
                save_plot(fig, 'theme_clusters.png')
            else:
                print("Skipping cluster plot - required columns (x, y, cluster) not found")
        except Exception as e:
            print(f"Error generating cluster plot: {str(e)}")

if __name__ == '__main__':
    main()
