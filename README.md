
# 🛍️ Product Recommendation System (KNN-Based)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-KNN-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Colab](https://img.shields.io/badge/Google-Colab-yellow)

**Item-Based Collaborative Filtering System using K-Nearest Neighbors Algorithm**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/product-recommendation-system/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app/)

## 📋 Overview

A complete machine learning pipeline for product recommendations that:
- Processes product data (titles, descriptions, categories, ratings, prices)
- Trains KNN model with combined feature engineering
- Evaluates recommendation quality with multiple metrics
- Exports trained model for production use
- Includes ready-to-deploy web application

## 🚀 Quick Start

### Google Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/product-recommendation-system/blob/main/Product_Recommendation_System.ipynb)

1. Upload your product dataset
2. Run all cells sequentially
3. Get trained model and recommendations
4. Download model files automatically

📊 Features
Core ML Pipeline
  - ✅ Data Preprocessing - Text cleaning, missing value handling
  - ✅ Feature Engineering - TF-IDF vectors + numerical features
  - ✅ Model Training - KNN with cosine similarity
  - ✅ Evaluation - Similarity scores, category accuracy
  - ✅ Export - Model persistence with Pickle

Output Files
- product_recommendation_model.pkl - Trained model + vectorizer
- product_recommendations.csv - All recommendation pairs
- model_summary.txt - Performance report

🏗️ Architecture

Raw Data → Preprocessing → Feature Engineering → KNN Training → Model Evaluation
                                                                     
   CSV File   →   Cleaning & Encoding  → TF-IDF + Numerical Scaling → Cosine Similarity Calculation  → Metrics Calculation   
            
   
📈 Model Performance
Metric	Score	Description

- Avg Similarity	0.85+	How similar recommendations are to source
- Category Match	80%+	Recommendations in same category
- Price Similarity	70%+	Recommendations in similar price range

# Get recommendations for a product
product_id = 8376765  # Backpack
recommendations = get_recommendations(product_id, n=5)

- Output similar products with:
- Titles, categories, ratings
- Similarity scores (0-1)
- Prices and product IDs
 
Features:
- Search by Product ID or Name
- Real-time recommendations
- Similarity scores visualization
- Product details display

🔧 Technologies Used
- Python 3.8+
- Scikit-learn - KNN, TF-IDF, Cosine Similarity
- Pandas & NumPy - Data manipulation
- Matplotlib & Seaborn - Visualization
- Pickle - Model serialization

📚 Dataset Requirements
The system expects a CSV file with columns:
- product_id - Unique identifier
- title - Product name
- product_description - Product description
- rating - Product rating (0-5)
- initial_price - Product price
- category - Product category
- product_details - Additional details (optional)
- product_specifications - Technical specs (optional)

🎓 How It Works
- Text Processing: Convert titles/descriptions to TF-IDF vectors
- Feature Combination: Merge with normalized ratings/prices
- Similarity Calculation: Use cosine similarity in KNN
- Recommendation Generation: Find nearest neighbors
- Quality Evaluation: Calculate accuracy metrics


📁 Project Structure
AI-Based-Personalized-Product-Recommendation-System/
- E-Commerce_Dataset.csv              # Main product dataset
- Product_Recommendation_Training.ipynb  # Model training notebook
- product_recommendation_model.pkl     # Trained KNN model
- app.py                              # Streamlit web application
- product_recommendations.csv         # Generated recommendations
- requirements.txt                        # Main project dependencies
- README.md                              # This file

Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

Option 1: Quick Installation (Recommended)
1. Clone the repository
- git clone https://github.com/yourusername/ai-recommendation-project-pcp.git
- cd ai-recommendation-project-pcp

2. Install dependencies
- pip install -r requirements.txt

3. Run the web application
- cd webapp
- streamlit run app.py

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
