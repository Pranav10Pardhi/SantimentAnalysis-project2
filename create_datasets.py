import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sentiment analysis dataset
sentiment_data = {
    'text': [
        "This product is amazing! Great customer service.",
        "Terrible experience, would not recommend.",
        "Pretty good product, but delivery was slow",
        "Love the features, best purchase ever!",
        "Waste of money, very disappointed",
        "The customer support team was very helpful",
        "Average product, nothing special",
        "Outstanding quality and value for money",
        "Product arrived damaged, poor packaging",
        "Really impressed with the performance"
    ],
    'sentiment': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # 1 for positive, 0 for negative
}
sentiment_df = pd.DataFrame(sentiment_data)
sentiment_df.to_csv('c:/Users/HP/OneDrive/Desktop/Notes/SentimentAnalysis/sentiment_data.csv', index=False)

# Create customer segmentation dataset
n_customers = 500
customer_data = {
    'CustomerID': range(1, n_customers + 1),
    'RecencyDays': np.random.randint(1, 365, n_customers),
    'FrequencyOrders': np.random.randint(1, 50, n_customers),
    'MonetaryValue': np.random.uniform(100, 5000, n_customers),
    'AverageOrderValue': np.random.uniform(50, 500, n_customers)
}
customer_df = pd.DataFrame(customer_data)
customer_df.to_csv('c:/Users/HP/OneDrive/Desktop/Notes/SentimentAnalysis/customer_data.csv', index=False)

# Create sales dataset
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
sales_data = {
    'Date': dates,
    'Sales': np.random.normal(10000, 2000, len(dates)),
    'Orders': np.random.randint(50, 200, len(dates)),
    'CustomerSatisfaction': np.random.uniform(3.5, 5.0, len(dates)),
    'ROI': np.random.uniform(0.15, 0.25, len(dates))
}
sales_df = pd.DataFrame(sales_data)
sales_df['Sales'] = sales_df['Sales'].apply(lambda x: max(x, 0))  # Ensure no negative sales
sales_df.to_csv('c:/Users/HP/OneDrive/Desktop/Notes/SentimentAnalysis/sales_data.csv', index=False)

print("Datasets created successfully!")