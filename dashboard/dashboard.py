import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to load data
@st.cache_data
def load_data():
    day_df = pd.read_csv('dashboard/day.csv')
    hour_df = pd.read_csv('dashboard/hour.csv')
    return day_df, hour_df

# Load the dataset
day_df, hour_df = load_data()

# Dashboard title
st.title("Bike Rental Data Analysis")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Sidebar: Date range filter
st.sidebar.header("Bike Rental Dashboard")
st.sidebar.image("https://img.freepik.com/free-vector/commuting-by-bike-concept-illustration_114360-28232.jpg?w=826&t=st=1727330117~exp=1727330717~hmac=eaebd8f9d38372e840f921e4e2823a8e67ddd29bad22112ada47d22dc7c6ca35")

day_df['date'] = pd.to_datetime(day_df['date'])

min_date = day_df['date'].min().date()
max_date = day_df['date'].max().date()

day_df['date'] = pd.to_datetime(day_df['date'])
hour_df['date'] = pd.to_datetime(hour_df['date'])

# Date range input with fallback for end date
date_range = st.sidebar.date_input(
    "Select date range:",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Ensure both start and end dates are selected, else use max_date for end date
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range[0]
    end_date = max_date

# Filter the data by the selected date range
filtered_day_df = day_df[(day_df['date'] >= pd.to_datetime(start_date)) & (day_df['date'] <= pd.to_datetime(end_date))]
filtered_hour_df = hour_df[(hour_df['date'] >= pd.to_datetime(start_date)) & (hour_df['date'] <= pd.to_datetime(end_date))]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Section: Daily Report
st.subheader('Daily Rentals')

daily_report = filtered_day_df.groupby(filtered_day_df['date'].dt.date)['count'].sum().reset_index()
daily_report.columns = ['Date', 'Total Rentals']

# Daily metrics
col1, col2 = st.columns(2)

with col1:
    total_rentals = daily_report['Total Rentals'].sum()
    st.metric("Total Rentals", value=total_rentals)

with col2:
    average_rentals = daily_report['Total Rentals'].mean()
    st.metric("Average Daily Rentals", value=f"{average_rentals:.2f}")

# Plot daily rentals
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_report["Date"],
    daily_report["Total Rentals"],
    marker='o',
    linewidth=2,
    color="#90CAF9"
)
ax.set_title('Total Rentals per Day', fontsize=16)
ax.set_xlabel('Date', fontsize=15)
ax.set_ylabel('Total Rentals', fontsize=15)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)

st.pyplot(fig)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Section: Rentals by Season (Registered and Casual)
st.header("☀️ Rentals by Season (Registered and Casual)")
season_order = ["Spring", "Summer", "Fall", "Winter"]
filtered_day_df["season"] = pd.Categorical(filtered_day_df["season"], categories=season_order, ordered=True)
melted_df = pd.melt(filtered_day_df, id_vars="season", value_vars=["registered", "casual"], var_name="user_type", value_name="rentals")
season_plot_df = melted_df.groupby(["season", "user_type"], observed=False, as_index=False).sum()

colors_ = ["#FF9999", "#66B2FF"]
fig_season, ax = plt.subplots(figsize=(10, 4))
sns.barplot(y="rentals", x="season", hue="user_type", data=season_plot_df, palette=colors_, ax=ax)
ax.set_title("Number of Rentals by User Type and Season", fontsize=15)
ax.set_ylabel("Total Rentals")
ax.set_xlabel("Season")
st.pyplot(fig_season)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Section: Best Weather Condition
st.header("☁️ Best Weather Condition for Rentals")
weather_plot_df = filtered_hour_df.groupby("weather_condition", observed=False)['count'].sum().reset_index()
weather_plot_df = weather_plot_df.sort_values(by='count', ascending=False)
weather_plot_df['weather_condition'] = weather_plot_df['weather_condition'].replace({
    'Light_precipitation': 'Light Weather',
    'Heavy_precipitation': 'Heavy Weather'
})

colors_weather = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
fig_weather, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='weather_condition', y='count', data=weather_plot_df, palette=colors_weather, ax=ax, hue="weather_condition")
ax.set_title("Number of Rentals by Weather Condition", fontsize=15)
ax.set_ylabel("Total Rentals")
ax.set_xlabel("Weather Condition")
st.pyplot(fig_weather)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Section: Correlation Between Temperature and Rentals
st.header("🌡️ Correlation Between Temperature and Rentals")
correlation_matrix = filtered_day_df[['temperature', 'feels_temperature', 'count']].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

fig_heatmap, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
ax.set_title("Correlation of Temperature with Count")
st.pyplot(fig_heatmap)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Section: Clustering Based on Windspeed
st.header("🍃 Clustering Based on Windspeed")
filtered_hour_df['windspeed_category'] = pd.cut(filtered_hour_df['windspeed'], bins=[-0.1, 0.2, 0.4, 1.1], labels=['Low', 'Medium', 'High'])
windspeed_cluster = filtered_hour_df.groupby('windspeed_category', observed=False)['count'].sum().reset_index()

fig_windspeed, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='windspeed_category', y='count', data=windspeed_cluster, palette='coolwarm', ax=ax, hue="windspeed_category")
ax.set_title('Total Rentals by Windspeed Category', fontsize=16)
ax.set_xlabel('Windspeed Category', fontsize=12)
ax.set_ylabel('Total Rentals', fontsize=12)
st.pyplot(fig_windspeed)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
