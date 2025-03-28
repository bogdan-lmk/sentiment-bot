from setuptools import setup, find_packages

setup(
    name="ai-santiment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "textblob==0.17.1",
        "pandas==2.0.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "schedule==1.2.0",
        "spacy==3.7.2",
        "transformers==4.36.2",
        "scikit-learn==1.3.2",
        "plotly==5.18.0",
        "requests==2.31.0",
        "telethon==1.28.5",
        "openai==1.12.0",
        "aiogram==3.0.0b7",
        "streamlit==1.29.0",
        "python-dotenv==1.0.0",
        "fpdf2==2.7.7"
    ],
    python_requires=">=3.9",
)
