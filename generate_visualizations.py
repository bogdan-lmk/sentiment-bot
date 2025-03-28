import os
import sys
import pandas as pd
from src.visualization.charts import DataVisualizer

def generate_visualizations():
    try:
        # Check matplotlib installation
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Error: matplotlib is not installed. Please install it with: pip install matplotlib")
            return

        # Create output directory
        os.makedirs("reports/visualizations", exist_ok=True)

        # Check if data files exist
        required_files = [
            "data/processed/sentiment_analysis.csv",
            "data/processed/keywords.csv", 
            "data/processed/message_trends.csv",
            "data/processed/message_clusters.csv"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                print(f"Error: Required data file not found: {file}")
                return

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
            return True
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            return False

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    generate_visualizations()
