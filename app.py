import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="E-Commerce Product Recommendations",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-card {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
        transition: transform 0.3s;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .product-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1F2937;
        margin-bottom: 10px;
    }
    .product-price {
        color: #059669;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .product-rating {
        color: #F59E0B;
        font-weight: bold;
    }
    .similarity-badge {
        background-color: #3B82F6;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# Load model and data
@st.cache_resource
def load_model():
    """Load the trained recommendation model"""
    try:
        with open('product_recommendation_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_data():
    """Load the product dataset"""
    try:
        df = pd.read_csv('E-Commerce_Dataset.csv')
        # Clean the data
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
        df['initial_price'] = pd.to_numeric(df['initial_price'], errors='coerce').fillna(0)
        df['title'] = df['title'].fillna('Unknown Product')
        df['category'] = df['category'].fillna('Uncategorized')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Load data
model_data = load_model()
df = load_data()

if model_data is None or df is None:
    st.error("Failed to load model or data. Please check the files.")
    st.stop()

# Extract model components
knn_model = model_data['knn_model']
combined_features = model_data['combined_features']


# Helper functions
def get_product_image(product_id):
    """Get product image from dataset"""
    try:
        product = df[df['product_id'] == product_id]
        if len(product) > 0:
            images = product.iloc[0].get('images', '')
            if pd.notna(images) and images != '':
                # Extract first image URL
                image_urls = images.split(',')
                if image_urls:
                    return image_urls[0]
    except:
        pass
    return None


def display_product_card(product, show_buy_btn=True):
    """Display a product card"""
    # Display product image
    image_url = get_product_image(product['product_id'])
    if image_url:
        try:
            response = requests.get(image_url, timeout=5)
            img = Image.open(BytesIO(response.content))
            st.image(img, width=150)
        except:
            st.image("https://via.placeholder.com/150x150/CCCCCC/000000?text=No+Image", width=150)
    else:
        st.image("https://via.placeholder.com/150x150/CCCCCC/000000?text=No+Image", width=150)

    st.markdown(f"<div class='product-title'>{product['title'][:50]}...</div>", unsafe_allow_html=True)

    # Rating stars
    rating = product.get('rating', 0)
    stars = "⭐" * int(rating) + "☆" * (5 - int(rating))
    st.markdown(f"<div class='product-rating'>{stars} ({rating})</div>", unsafe_allow_html=True)

    # Price
    price = product.get('initial_price', 0)
    st.markdown(f"<div class='product-price'>₹{price:,.0f}</div>", unsafe_allow_html=True)

    # Category
    category = product.get('category', 'Uncategorized')
    st.caption(f"Category: {category}")

    # Product ID
    st.caption(f"ID: {product['product_id']}")

    # Buy button
    if show_buy_btn:
        if st.button(f"🛒 Buy Now", key=f"buy_{product['product_id']}", use_container_width=True):
            st.session_state.selected_product = product['product_id']
            st.success(f"Added to cart: {product['title'][:30]}...")
            st.rerun()


def get_recommendations(product_id, n=5):
    """Get recommendations for a product"""
    try:
        # Find product index
        product_idx = df[df['product_id'] == product_id].index[0]

        # Get similar products
        distances, indices = knn_model.kneighbors(
            combined_features[product_idx].reshape(1, -1),
            n_neighbors=n + 1
        )

        # Exclude the product itself
        similar_indices = indices[0][1:]
        similar_distances = distances[0][1:]

        recommendations = []
        for idx, dist in zip(similar_indices, similar_distances):
            similar_product = df.iloc[idx]
            similarity = 1 - dist

            recommendations.append({
                'product_id': similar_product['product_id'],
                'title': similar_product['title'],
                'category': similar_product.get('category', 'N/A'),
                'rating': similar_product.get('rating', 0),
                'price': similar_product.get('initial_price', 0),
                'similarity_score': round(similarity * 100, 1)  # Percentage
            })

        return recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []


# Main App
def main():
    # Header
    st.markdown("<h1 class='main-header'>🛍️ E-Commerce Product Recommendations</h1>", unsafe_allow_html=True)

    # Initialize session state
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None

    # Main content area - Products Grid
    st.header("📦 All Products")
    
    # Pagination
    products_per_page = 10
    total_pages = (len(df) - 1) // products_per_page + 1
    
    # Page selector
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.selectbox("Select Page", range(1, total_pages + 1), key="page_selector")
    
    start_idx = (page - 1) * products_per_page
    end_idx = start_idx + products_per_page
    page_df = df.iloc[start_idx:end_idx]
    
    st.write(f"Showing products {start_idx + 1} to {min(end_idx, len(df))} of {len(df)} total products")

    # Display products in 5-column grid
    cols = st.columns(5)
    for idx, (_, product) in enumerate(page_df.iterrows()):
        with cols[idx % 5]:
            display_product_card(product)

    # Recommendations Section (Only shows when a product is bought/added to cart)
    if st.session_state.selected_product is not None:
        st.markdown("---")
        st.markdown("## 🛒 You might also like these products")
        
        # Get selected product details
        selected_product = df[df['product_id'] == st.session_state.selected_product]
        if len(selected_product) > 0:
            selected_product = selected_product.iloc[0]

            # Display selected product info
            st.info(f"**Based on your purchase:** {selected_product['title']} (₹{selected_product['initial_price']:,})")
            
            # Clear recommendations button
            if st.button("❌ Clear Recommendations", use_container_width=False):
                st.session_state.selected_product = None
                st.rerun()
            
            # Get recommendations using the ML model
            with st.spinner("🤖 Finding similar products you might like..."):
                recommendations = get_recommendations(st.session_state.selected_product, 10)

            if recommendations:
                st.success(f"Here are {len(recommendations)} products similar to your purchase!")
                
                # Display recommendations in grid
                rec_cols = st.columns(5)
                for i, rec in enumerate(recommendations):
                    with rec_cols[i % 5]:
                        # Display recommendation
                        image_url = get_product_image(rec['product_id'])
                        if image_url:
                            try:
                                response = requests.get(image_url, timeout=5)
                                img = Image.open(BytesIO(response.content))
                                st.image(img, use_column_width=True)
                            except:
                                st.image("https://via.placeholder.com/150x150/CCCCCC/000000?text=No+Image")
                        else:
                            st.image("https://via.placeholder.com/150x150/CCCCCC/000000?text=No+Image")

                        st.markdown(f"**{rec['title'][:40]}...**")
                        
                        # Similarity badge
                        st.markdown(f"<span class='similarity-badge'>🎯 {rec['similarity_score']}% Match</span>", 
                                  unsafe_allow_html=True)
                        
                        st.write(f"⭐ {rec['rating']} | ₹{rec['price']:,}")
                        st.caption(f"Category: {rec['category']}")

                        # Buy button for recommendations
                        if st.button("🛒 Buy This Too", key=f"rec_buy_{rec['product_id']}", use_container_width=True):
                            st.session_state.selected_product = rec['product_id']
                            st.success(f"Added to cart: {rec['title'][:30]}...")
                            st.rerun()
            else:
                st.warning("No similar products found.")
        else:
            st.error("Product not found!")
            st.session_state.selected_product = None
            
if __name__ == "__main__":
    main()