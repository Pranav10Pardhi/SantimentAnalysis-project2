import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("📊 AI Dashboard – Sentiment, Segmentation & Sales")

# ---------------- SAFE FILE READER ----------------
def safe_read_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(uploaded_file, encoding="latin1")
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload CSV or Excel.")
            return None
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None

# ---------------- DOWNLOAD HELPER ----------------
def convert_df(df):
    output = BytesIO()
    df.to_excel(output, index=False, engine="xlsxwriter")
    processed_data = output.getvalue()
    return processed_data

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Customer Segmentation", "Sales Dashboard"])

# ---------------- SENTIMENT ANALYSIS ----------------
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

            # Probabilities
            probs = {
                "Positive": max(sentiment, 0),
                "Negative": max(-sentiment, 0),
                "Neutral": 1 - abs(sentiment)
            }
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}

            prob_df = pd.DataFrame({
                "Sentiment": probs.keys(),
                "Probability": [f"{v:.2f} ({v*100:.0f}%)" for v in probs.values()]
            })

            st.table(prob_df)

# ---------------- CUSTOMER SEGMENTATION ----------------
elif page == "Customer Segmentation":
    st.header("👥 Customer Segmentation")
    uploaded_file = st.file_uploader("Upload Customer Data (CSV or Excel)", type=["csv", "xls", "xlsx"])

    if uploaded_file:
        data = safe_read_file(uploaded_file)
        if data is not None:
            st.subheader("Dataset Preview")
            st.dataframe(data.head())

            if "CustomerID" in data.columns and "Segment" in data.columns:
                seg_counts = data["Segment"].value_counts().reset_index()
                seg_counts.columns = ["Segment", "Count"]

                fig = px.pie(seg_counts, names="Segment", values="Count", title="Customer Distribution by Segment")
                st.plotly_chart(fig)

                st.download_button(
                    label="📥 Download Segmentation Data",
                    data=convert_df(seg_counts),
                    file_name="segmentation_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ Dataset must include 'CustomerID' and 'Segment' columns.")

# ---------------- SALES DASHBOARD ----------------
elif page == "Sales Dashboard":
    st.header("💰 Sales Dashboard")
    uploaded_file = st.file_uploader("Upload Sales Data (CSV or Excel)", type=["csv", "xls", "xlsx"])

    if uploaded_file:
        data = safe_read_file(uploaded_file)
        if data is not None:
            st.subheader("Dataset Preview")
            st.dataframe(data.head())

            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
                data = data.dropna(subset=["Date"])

                # Date Filter
                date_range = st.date_input(
                    "Select Date Range",
                    [data["Date"].min().date(), data["Date"].max().date()]
                )
                filtered = data[(data["Date"].dt.date >= date_range[0]) &
                                (data["Date"].dt.date <= date_range[1])]

                col1, col2, col3 = st.columns(3)
                with col1:
                    total_sales = filtered["Sales"].sum() if "Sales" in filtered.columns else 0
                    st.metric("Total Sales", f"${total_sales:,.2f}")
                with col2:
                    avg_orders = filtered["Orders"].mean() if "Orders" in filtered.columns else 0
                    st.metric("Average Daily Orders", f"{avg_orders:.0f}")
                with col3:
                    roi = filtered["ROI"].mean() if "ROI" in filtered.columns else 0
                    st.metric("Average ROI", f"{roi:.1%}")

                if "Sales" in filtered.columns:
                    fig = px.line(filtered, x="Date", y="Sales", title="Sales Trend Over Time")
                    st.plotly_chart(fig)

                st.download_button(
                    label="📥 Download Filtered Sales Data",
                    data=convert_df(filtered),
                    file_name="sales_filtered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ Dataset must include a 'Date' column.")
