# Time Series Forecasting of Electricity Consumption

## Overview

This project aims to forecast the hourly electricity consumption for different buildings on the Michigan State University (MSU) campus. Electricity consumption data is recorded every hour, and our goal is to predict future usage for different forecast windows (1 hour, 10 hours, 24 hours, etc.). 

We have tried:
- **Naive Forecasting** for 1-hour ahead predictions
- **ARIMA (AutoRegressive Integrated Moving Average)** for 10-hour and 24-hour ahead predictions, with parameters found using an **auto ARIMA** approach (e.g., `(p, d, q) = (1, 1, 2)`)

So far, our results indicate that the Naive model surprisingly outperforms more complex models for the 1-hour forecast window. For longer windows, ARIMA models have been employed with reasonable but not exceptional performance.

## Repository Contents

1. **`NaiveForecasting.ipynb`**  
   Demonstrates the Naive approach for 1-hour ahead predictions and compares predictions with actual consumption values.

2. **`ARIMA-10hours.ipynb`**  
   Explores ARIMA models for 10-hour ahead predictions using auto ARIMA to select optimal parameters. Plots and evaluation metrics are provided.

3. **Figures/Plots**  
   - Plots showing actual vs. predicted consumption values for the test set, comparing Naive Forecasting and ARIMA approaches for different time horizons.

## Data Description

- **Source**: Hourly electricity consumption data from MSU campus buildings.
- **Frequency**: Hourly (every 60 minutes).
- **Features**:  
  - **Timestamp**: Date/Time of recording.  
  - **Consumption**: Electricity usage in MWh (or other specified units).  
  - *(Optional)* Additional weather or contextual data can be merged if available (temperature, humidity, occupancy, etc.).

## Project Workflow

1. **Data Preprocessing**  
   - Load the raw data and handle missing values if any.  
   - Convert timestamps to a uniform date-time format.  
   - Resample or aggregate data if needed.  
   - Split data into training and test sets (often using a time-based split).

2. **Exploratory Data Analysis (EDA)**  
   - Visualize trends, seasonality, and outliers.  
   - Check for daily, weekly, or monthly patterns.  
   - Identify correlations with external factors if applicable (e.g., temperature).

3. **Modeling**  
   - **Naive Forecasting**: Use the last observed value as the forecast for the next step (especially for 1-hour ahead).  
   - **ARIMA**: Use an Auto ARIMA approach to determine optimal `(p, d, q)` parameters for multi-hour ahead forecasting (10-hour, 24-hour, etc.).  
   - **Evaluation Metrics**: Compare predictions vs. actual values using metrics such as MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), or MAPE (Mean Absolute Percentage Error).

4. **Results**  
   - Visualize predicted vs. actual consumption.  
   - Summarize metrics for each approach and forecast horizon.  
   - So far, the Naive model has given surprisingly strong results for 1-hour forecasts, while ARIMA is used for longer horizons with moderate success.

5. **Future Steps**  
   - Implement and compare additional forecasting methods (see [Suggestions](#suggestions-for-further-methods))  
   - Fine-tune hyperparameters for ARIMA or other models  
   - Investigate additional features (weather, occupancy, day-of-week, etc.) for better predictive performance  

## Installation and Usage

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/TimeSeriesForecasting.git
   cd TimeSeriesForecasting
   ```

2. **Set Up the Environment**  
   - Create a virtual environment (optional but recommended):  
     ```bash
     python -m venv venv
     source venv/bin/activate  # Linux/Mac
     venv\Scripts\activate     # Windows
     ```
   - Install dependencies (example `requirements.txt`):  
     ```bash
     pip install -r requirements.txt
     ```
   - Typical libraries used: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `pmdarima`, `scikit-learn`, `jupyter`, etc.

3. **Run the Notebooks**  
   - Launch Jupyter Notebook or JupyterLab:  
     ```bash
     jupyter notebook
     ```
   - Open `NaiveForecasting.ipynb` or `ARIMA-10hours.ipynb` to reproduce the analyses and plots.

## Results

- **Naive Forecast (1-hour)**:  
  - Often outperforms other methods for very short-term forecasts, likely because hourly consumption patterns can be relatively stable in consecutive hours.

- **ARIMA (10-hour / 24-hour)**:  
  - Provides moderate accuracy for multi-step forecasts.  
  - Parameter tuning was done with Auto ARIMA, resulting in `(p, d, q) = (1, 1, 2)` for the data tested.  
  - Overall error metrics are higher than the Naive approach for single-step, but better for multi-step horizons compared to a naive multi-step extension.

## Suggestions for Further Methods

Below are some additional methods and techniques you could explore to further improve or compare against your current ARIMA-based approach:

1. **Seasonal ARIMA (SARIMA)**  
   - If the data exhibits strong daily or weekly seasonality, a Seasonal ARIMA model (`SARIMA` or `SARIMAX`) might capture those patterns better than a non-seasonal ARIMA.

2. **Exponential Smoothing Methods (Holt-Winters)**  
   - Useful for capturing trends and seasonality. Holt-Winters can be quite effective if you observe a clear seasonal cycle (e.g., daily patterns in electricity usage).

3. **Prophet (Facebook Prophet)**  
   - A popular library for time series forecasting, especially when data has multiple seasonality patterns (daily, weekly, yearly). Easy to set up and often gives decent performance out-of-the-box.

4. **Machine Learning Regression Approaches**  
   - **Random Forest Regressor** or **Gradient Boosted Trees** (e.g., XGBoost, LightGBM, CatBoost).  
   - Transform your time series into a supervised learning problem by creating lag features, rolling means, day-of-week, hour-of-day, etc.  
   - These models can capture non-linearities and interactions more easily than ARIMA.

5. **Neural Networks**  
   - **Recurrent Neural Networks (RNNs)** such as **LSTM** or **GRU** can capture long-term dependencies and patterns in sequential data.  
   - **Temporal Convolutional Networks (TCN)** or **Transformers** adapted for time series can also be explored.

6. **Hybrid / Ensemble Methods**  
   - Combine forecasts from multiple models to potentially achieve more robust performance (e.g., average or weighted average of ARIMA, Prophet, and a neural network).

7. **Feature Engineering**  
   - Incorporate external factors: weather (temperature, humidity), calendar events (holidays, weekends, etc.), building occupancy or event schedules.  
   - Generate time-based features (day of week, hour of day, month, etc.) if using machine learning methods.

8. **Multi-step Forecasting Strategies**  
   - **Direct forecasting**: Train separate models for each forecast horizon (e.g., 1-hour ahead, 2-hour ahead, …).  
   - **Iterative forecasting**: Use the model’s predictions as inputs for subsequent forecasts.  
   - **Recursive or Hybrid approaches**: Combine the above strategies.

Exploring these methods will allow you to compare performance against ARIMA and select the best approach (or combination of approaches) for your specific forecasting horizons.

## License

This project is licensed under the [MIT License](LICENSE) (or whichever license you choose). Please see the `LICENSE` file for details.

---