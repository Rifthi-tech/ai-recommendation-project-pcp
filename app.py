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


def display_product_card(product, show_recommend_btn=True):
    """Display a product card"""
    col1, col2 = st.columns([1, 3])

    with col1:
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

    with col2:
        st.markdown(f"<div class='product-title'>{product['title']}</div>", unsafe_allow_html=True)

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
        st.caption(f"Product ID: {product['product_id']}")

        # Recommend button
        if show_recommend_btn:
            if st.button(f"Get Similar Products", key=f"btn_{product['product_id']}"):
                st.session_state.selected_product = product['product_id']
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

    # Sidebar
    with st.sidebar:
        st.header("🔍 Search & Filter")

        # Category filter
        categories = ['All'] + sorted(df['category'].unique().tolist())
        selected_category = st.selectbox("Filter by Category", categories)

        # Price range filter
        min_price = float(df['initial_price'].min())
        max_price = float(df['initial_price'].max())
        price_range = st.slider(
            "Price Range (₹)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )

        # Rating filter
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.0, 0.5)

        st.markdown("---")
        st.header("📊 Statistics")
        st.write(f"Total Products: {len(df)}")
        st.write(f"Categories: {len(df['category'].unique())}")
        st.write(f"Avg Rating: {df['rating'].mean():.2f}")

        # Search by ID
        st.markdown("---")
        st.header("🔎 Search Product")
        search_id = st.number_input("Enter Product ID", min_value=int(df['product_id'].min()),
                                    max_value=int(df['product_id'].max()), value=8376765)
        if st.button("Search by ID"):
            st.session_state.selected_product = int(search_id)
            st.rerun()

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        # Product Grid Section
        st.header("📦 Available Products")

        # Filter products
        filtered_df = df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]

        filtered_df = filtered_df[
            (filtered_df['initial_price'] >= price_range[0]) &
            (filtered_df['initial_price'] <= price_range[1]) &
            (filtered_df['rating'] >= min_rating)
            ]

        # Display products in grid
        cols = st.columns(3)
        for idx, product in filtered_df.head(18).iterrows():  # Show first 18 products
            with cols[idx % 3]:
                with st.container():
                    st.markdown("<div class='product-card'>", unsafe_allow_html=True)
                    display_product_card(product)
                    st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Recommendations Section
        st.header("🎯 Quick Recommendations")

        # Show popular products
        st.subheader("🔥 Popular Products")
        popular_products = df.nlargest(5, 'rating')
        for _, product in popular_products.iterrows():
            with st.container():
                st.markdown(f"**{product['title'][:30]}...**")
                st.write(f"⭐ {product['rating']} | ₹{product['initial_price']:,}")
                if st.button("Similar", key=f"pop_{product['product_id']}"):
                    st.session_state.selected_product = product['product_id']
                    st.rerun()
                st.divider()

    # Recommendations Display Section (Full Width)
    if st.session_state.selected_product is not None:
        st.markdown("---")
        st.header("🎯 Product Recommendations")

        # Get selected product details
        selected_product = df[df['product_id'] == st.session_state.selected_product]
        if len(selected_product) > 0:
            selected_product = selected_product.iloc[0]

            # Display selected product
            st.subheader("Selected Product:")
            col1, col2 = st.columns([1, 3])

            with col1:
                image_url = get_product_image(selected_product['product_id'])
                if image_url:
                    try:
                        response = requests.get(image_url, timeout=5)
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=200)
                    except:
                        st.image("https://via.placeholder.com/200x200/CCCCCC/000000?text=No+Image", width=200)

            with col2:
                st.markdown(f"### {selected_product['title']}")
                st.write(f"**Category:** {selected_product.get('category', 'N/A')}")
                st.write(f"**Rating:** ⭐ {selected_product.get('rating', 'N/A')}")
                st.write(f"**Price:** ₹{selected_product.get('initial_price', 'N/A'):,}")
                st.write(f"**Product ID:** {selected_product['product_id']}")

            # Get recommendations
            recommendations = get_recommendations(st.session_state.selected_product, 6)

            if recommendations:
                st.subheader(f"Similar Products ({len(recommendations)} found):")

                # Display recommendations in grid
                rec_cols = st.columns(3)
                for i, rec in enumerate(recommendations):
                    with rec_cols[i % 3]:
                        with st.container():
                            st.markdown("<div class='product-card'>", unsafe_allow_html=True)

                            # Display recommendation
                            image_url = get_product_image(rec['product_id'])
                            if image_url:
                                try:
                                    response = requests.get(image_url, timeout=5)
                                    img = Image.open(BytesIO(response.content))
                                    st.image(img, use_column_width=True)
                                except:
                                    st.image("https://via.placeholder.com/150x150/CCCCCC/000000?text=No+Image")

                            st.markdown(f"**{rec['title'][:40]}...**")
                            st.write(f"Similarity: {rec['similarity_score']}%")
                            st.write(f"⭐ {rec['rating']} | ₹{rec['price']:,}")

                            # Add to cart button
                            if st.button("View Details", key=f"rec_{rec['product_id']}"):
                                st.session_state.selected_product = rec['product_id']
                                st.rerun()

                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this product.")

        # Clear selection button
        if st.button("Clear Selection"):
            st.session_state.selected_product = None
            st.rerun()

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.write("Powered by KNN Machine Learning")
        st.write(f"Total Products in System: {len(df)}")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()