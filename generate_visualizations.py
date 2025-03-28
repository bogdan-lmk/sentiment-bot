import os
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.charts import DataVisualizer

def generate_visualizations():
    # Create output directory
    os.makedirs("reports/visualizations", exist_ok=True)

    # Load data
    sentiment_df = pd.read_csv("data/processed/sentiment_analysis.csv")
    keywords_df = pd.read_csv("data/processed/keywords.csv")
    trends_df = pd.read_csv("data/processed/message_trends.csv")
    clusters_df = pd.read_csv("data/processed/message_clusters.csv")

    # Initialize visualizer
    visualizer = DataVisualizer(sentiment_df)

    # Generate and save plots
    try:
        fig = visualizer.plot_sentiment_distribution()
        fig.savefig("reports/visualizations/sentiment_distribution.png")
        plt.close(fig)
        
        fig = visualizer.plot_top_keywords(keywords_df)
        fig.savefig("reports/visualizations/top_keywords.png")
        plt.close(fig)
        
        fig = visualizer.plot_trends(trends_df)
        fig.savefig("reports/visualizations/message_trends.png")
        plt.close(fig)
        
        fig = visualizer.plot_clusters(clusters_df)
        fig.savefig("reports/visualizations/message_clusters.png")
        plt.close(fig)

        print("Visualizations generated successfully in reports/visualizations/")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    generate_visualizations()
