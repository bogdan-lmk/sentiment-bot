import os
import sys
import pandas as pd
from src.visualization.charts import DataVisualizer

def generate_visualizations(geo):
    try:
        # Check matplotlib installation
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Error: matplotlib is not installed. Please install it with: pip install matplotlib")
            return False

        # Create output directory
        vis_dir = os.path.join("reports", geo, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Check if data files exist
        required_files = [
            f"data/processed/{geo}/sentiment_analysis.csv",
            f"data/processed/{geo}/keywords.csv", 
            f"data/processed/{geo}/message_trends.csv",
            f"data/processed/{geo}/message_clusters.csv",
            f"data/processed/{geo}/needs_analysis.csv"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                print(f"Error: Required data file not found: {file}")
                return

        # Load data
        sentiment_df = pd.read_csv(f"data/processed/{geo}/sentiment_analysis.csv")
        keywords_df = pd.read_csv(f"data/processed/{geo}/keywords.csv")
        trends_df = pd.read_csv(f"data/processed/{geo}/message_trends.csv")
        clusters_df = pd.read_csv(f"data/processed/{geo}/message_clusters.csv")
        needs_df = pd.read_csv(f"data/processed/{geo}/needs_analysis.csv")

        # Initialize visualizer with sentiment data
        visualizer = DataVisualizer(sentiment_df)

        # Generate and save plots
        try:
            fig = visualizer.plot_sentiment_distribution()
            if fig is None:
                print("Skipping sentiment distribution plot - no valid data")
                return False
            fig.savefig(os.path.join(vis_dir, "sentiment_distribution.png"))
            plt.close(fig)
            
            fig = visualizer.plot_top_phrases(keywords_df)
            fig.savefig(os.path.join(vis_dir, "top_phrases.png"))
            plt.close(fig)
            
            fig = visualizer.plot_trends(trends_df)
            fig.savefig(os.path.join(vis_dir, "message_trends.png"))
            plt.close(fig)
            
            fig = visualizer.plot_needs_distribution(needs_df)
            if fig is not None:
                fig.savefig(os.path.join(vis_dir, "needs_distribution.png"))
                plt.close(fig)
            else:
                print("Skipping needs visualization - missing required data columns")
            
            fig = visualizer.plot_clusters(clusters_df)
            if fig is not None:  # Only save if we got a valid figure
                fig.savefig(os.path.join(vis_dir, "message_clusters.png"))
                plt.close(fig)
            else:
                print("Skipping cluster visualization - missing required data columns")

            print(f"Visualizations generated successfully in {vis_dir}/")
            return True
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            return False

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_visualizations.py <geo_code>")
        sys.exit(1)
    geo_code = sys.argv[1]
    generate_visualizations(geo_code)
