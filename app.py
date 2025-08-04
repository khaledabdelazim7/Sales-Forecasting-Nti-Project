import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

# Load model and columns
model = joblib.load('rf_model.pkl')
columns = joblib.load('model_columns.pkl')

# Optional: Load historical sales for visual comparison
try:
    df_sales = pd.read_csv("historical_sales.csv", parse_dates=['date'])
    df_sales['sales'] = np.expm1(df_sales['log_sales'])  # adjust if not in log
    has_sales_data = True
except:
    df_sales = pd.DataFrame()
    has_sales_data = False

st.title("📊 Sales Prediction")
st.write("Choose to predict by day or by month.")

# Choose prediction mode
mode = st.radio("🧭 Prediction Mode", options=["By Day", "By Month"])

if mode == "By Day":
    input_date = st.date_input("📅 Select a Date", value=datetime(2017, 7, 7))
    selected_dates = [input_date]
else:
    year = st.selectbox("📆 Select Year", list(range(2015, 2030)))
    month = st.selectbox("📅 Select Month", list(range(1, 13)))
    days_in_month = pd.Period(f'{year}-{month:02}').days_in_month
    selected_dates = [datetime(year, month, day) for day in range(1, days_in_month + 1)]

if st.button("🔮 Predict"):
    all_preds = []
    for date in selected_dates:
        # Prepare input features
        input_features = {}
        for col in columns:
            if 'Order_Year' in col:
                input_features[col] = date.year
            elif 'Order_Month' in col:
                input_features[col] = date.month
            elif 'Order_Day' in col:
                input_features[col] = date.day
            elif 'Order_Weekday' in col:
                input_features[col] = date.weekday()
            elif 'Ship_Year' in col:
                input_features[col] = date.year
            elif 'Ship_Month' in col:
                input_features[col] = date.month
            elif 'Ship_Day' in col:
                input_features[col] = min(date.day + 2, 28)
            elif 'Ship_Weekday' in col:
                input_features[col] = (date.weekday() + 2) % 7
            else:
                input_features[col] = 0  # default for other features

        input_df = pd.DataFrame([input_features])
        predicted_log = model.predict(input_df)[0]
        predicted_sales = np.expm1(predicted_log)
        all_preds.append((date, predicted_sales))

    # Output predictions
    if mode == "By Day":
        st.success(f"💰 Predicted Sales for {input_date}: **{all_preds[0][1]:.2f}**")
    else:
        st.subheader("📈 Predicted Sales for Each Day in Selected Month")
        for date, sale in all_preds:
            st.write(f"{date.strftime('%Y-%m-%d')}: **{sale:.2f}**")

        # Optional chart
        if has_sales_data:
            st.subheader("📉 Historical vs Predicted Sales (Visual)")
            df_plot = pd.DataFrame(all_preds, columns=['date', 'predicted_sales'])

            # Merge with historical data if overlapping
            if not df_sales.empty:
                merged = pd.merge(df_plot, df_sales, on='date', how='left')

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(merged['date'], merged['predicted_sales'], label='Predicted Sales', marker='o')
                ax.plot(merged['date'], merged['sales'], label='Actual Sales', linestyle='--', alpha=0.7)
                ax.set_title('📊 Sales Prediction vs Actual')
                ax.legend()
                st.pyplot(fig)
            else:
                st.line_chart(df_plot.set_index('date'))

