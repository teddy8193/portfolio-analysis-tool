# Portfolio Analysis Tool
## Abstract
This is a repository of the Portfolio Analysis Tool, which is designed to help people analyze stock performance.
The app retrieves stock data from Yahoo!Finance API.

## App
https://portfolio-analysis-tool.streamlit.app/

## How to use

![スクリーンショット 2024-09-08 002218](https://github.com/user-attachments/assets/a46e5bf3-1f10-4d51-8ef1-8aace3bb07a2)
1. Select the date range, add ticker symbols
2. Hit visualize button

## Plots
![スクリーンショット 2024-09-08 002235](https://github.com/user-attachments/assets/55b7cc44-3165-495c-9af6-e00219200718)
This plot visualizes the price (price + dividend if the option is enabled) change from the first day of the given range.

![スクリーンショット 2024-09-08 002252](https://github.com/user-attachments/assets/020e4256-9541-481c-99b9-fd9541a72682)
Yearly return and risk by day

![スクリーンショット 2024-09-08 002259](https://github.com/user-attachments/assets/f2c7ae62-726e-47a6-845b-5d31ccc9a759)
Correlation between given tickers

![スクリーンショット 2024-09-08 002316](https://github.com/user-attachments/assets/89faefe9-43e4-4d56-bc67-c4f7d2bcdf3c)
Efficient frontier plot with Sharpe Ratio by given tickers

## Tech stack
Python, Streamlit
