import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to calculate quarterly growth rate
def calculate_quarterly_growth_rate(data):
    data['Quarterly Growth Rate'] = data['Subscribers'].pct_change() * 100
    return data

# Function to calculate yearly growth rate
def calculate_yearly_growth_rate(data):
    data['Year'] = data['Time Period'].dt.year
    yearly_growth = data.groupby('Year')['Subscribers'].pct_change().fillna(0) * 100
    return yearly_growth

# Function to perform ARIMA forecasting
def arima_forecast(time_series, future_steps):
    differenced_series = time_series.diff().dropna()
    
    p, d, q = 1, 1, 1
    model = ARIMA(time_series, order=(p, d, q))
    results = model.fit()
    
    predictions = results.predict(len(time_series), len(time_series) + future_steps - 1)
    predictions = predictions.astype(int)
    
    forecast = pd.DataFrame({'Original': time_series, 'Predictions': predictions})
    
    return forecast

# Streamlit App
def main():
    st.title('Netflix Subscriptions Forecasting App')
    
    # Read data
    data = pd.read_csv('C:/Users/samab/second/Netflix-Subscriptions.csv')
    data['Time Period'] = pd.to_datetime(data['Time Period'], format='%d/%m/%Y')
    time_series = data.set_index('Time Period')['Subscribers']
    
    # Display original data plot
    st.subheader('Original Data')
    st.line_chart(time_series)
    
    # User input for future steps
    future_steps = st.number_input('Enter the number of future steps for forecasting:', min_value=1, value=10, step=1)
    
    # Perform ARIMA forecasting
    forecast = arima_forecast(time_series, future_steps)
    
    # Display forecasted data plot
    st.subheader('Forecasted Data')
    st.line_chart(forecast[['Original', 'Predictions']])
    
    # Display ACF and PACF plots
    st.subheader('Autocorrelation and Partial Autocorrelation Plots')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(time_series.diff().dropna(), ax=axes[0])
    plot_pacf(time_series.diff().dropna(), ax=axes[1])
    st.pyplot(fig)

    # Calculate and display Quarterly Growth Rate plot
    st.subheader('Netflix Quarterly Subscriptions Growth Rate')
    data = calculate_quarterly_growth_rate(data)
    quarterly_growth_fig = go.Figure()
    quarterly_growth_fig.add_trace(go.Bar(
        x=data['Time Period'],
        y=data['Quarterly Growth Rate'],
        marker_color=data['Quarterly Growth Rate'].apply(lambda x: 'green' if x > 0 else 'red'),
        name='Quarterly Growth Rate'
    ))
    quarterly_growth_fig.update_layout(xaxis_title='Time Period', yaxis_title='Quarterly Growth Rate (%)')
    st.plotly_chart(quarterly_growth_fig)

    # Calculate and display Yearly Subscriber Growth Rate plot
    st.subheader('Netflix Yearly Subscriber Growth Rate')
    yearly_growth = calculate_yearly_growth_rate(data)
    yearly_growth_fig = go.Figure()
    yearly_growth_fig.add_trace(go.Bar(
        x=data['Year'],
        y=yearly_growth,
        marker_color=yearly_growth.apply(lambda x: 'green' if x > 0 else 'red'),
        name='Yearly Growth Rate'
    ))
    yearly_growth_fig.update_layout(xaxis_title='Year', yaxis_title='Yearly Growth Rate (%)')
    st.plotly_chart(yearly_growth_fig)

if __name__ == '__main__':
    main()
