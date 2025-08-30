import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("📊 AI Dashboard – Sentiment, Segmentation & Sales")

# --- Safe CSV Reader ---
def safe_read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(uploaded_file, encoding="latin1")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Customer Segmentation", "Sales Dashboard"])

# --- Sentiment Analysis Page ---
if page == "Sentiment Analysis":
    st.header("📝 Sentiment Analysis")
    
    text = st.text_area("Enter text to analyze:")
    if st.button("Analyze Sentiment"):
        if text.strip():
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            if sentiment > 0:
                result = "Positive 😀"
            elif sentiment < 0:
                result = "Negative 😡"
            else:
                result = "Neutral 😐"
            
            st.success(f"Sentiment: **{result}**")
            
            # Probabilities (Normalized to 1.0)
            probs = {
                "Positive": max(sentiment, 0),
                "Negative": max(-sentiment, 0),
                "Neutral": 1 - abs(sentiment)
            }
            total = sum(probs.values())
            probs = {k: v/total for k,v in probs.items()}
            
            prob_df = pd.DataFrame({
                "Sentiment": probs.keys(),
                "Probability": [f"{v:.2f} ({v*100:.0f}%)" for v in probs.values()]
            })
            st.table(prob_df)

# --- Customer Segmentation Page ---
elif page == "Customer Segmentation":
    st.header("👥 Customer Segmentation")
    uploaded_file = st.file_uploader("Upload Customer Data CSV", type="csv")
    
    if uploaded_file:
        try:
            data = safe_read_csv(uploaded_file)
            st.subheader("Dataset Preview")
            st.dataframe(data.head())
            
            if 'CustomerID' in data.columns and 'Segment' in data.columns:
                seg_counts = data['Segment'].value_counts().reset_index()
                seg_counts.columns = ['Segment', 'Count']
                fig = px.pie(seg_counts, names='Segment', values='Count', title="Customer Distribution by Segment")
                st.plotly_chart(fig)
            else:
                st.warning("Dataset must include 'CustomerID' and 'Segment' columns.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# --- Sales Dashboard Page ---
elif page == "Sales Dashboard":
    st.header("💰 Sales Dashboard")
    uploaded_file = st.file_uploader("Upload Sales Data CSV", type="csv")
    
    if uploaded_file:
        try:
            data = safe_read_csv(uploaded_file)
            st.subheader("Dataset Preview")
            st.dataframe(data.head())
            
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'], errors="coerce")
                data = data.dropna(subset=['Date'])
                
                # Date Filter
                date_range = st.date_input("Select Date Range", 
                                           [data['Date'].min(), data['Date'].max()])
                filtered = data[(data['Date'].dt.date >= date_range[0]) &
                                (data['Date'].dt.date <= date_range[1])]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_sales = filtered['Sales'].sum() if 'Sales' in filtered.columns else 0
                    st.metric("Total Sales", f"${total_sales:,.2f}")
                with col2:
                    avg_orders = filtered['Orders'].mean() if 'Orders' in filtered.columns else 0
                    st.metric("Average Daily Orders", f"{avg_orders:.0f}")
                with col3:
                    roi = filtered['ROI'].mean() if 'ROI' in filtered.columns else 0
                    st.metric("Average ROI", f"{roi:.1%}")
                
                if 'Sales' in filtered.columns:
                    fig = px.line(filtered, x='Date', y='Sales', title="Sales Trend Over Time")
                    st.plotly_chart(fig)
            else:
                st.warning("Dataset must include a 'Date' column.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
