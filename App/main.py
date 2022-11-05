# pip install streamlit fbprophet yfinance plotly
from datetime import date
from modules import *
from plotly import graph_objs as go

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

url = 'https://raw.githubusercontent.com/Hrithik2212/Stock-Price-Prediction-App/master/App/Data/Indian-Stock-Market.csv'
stocks_df = pd.read_csv(url)
stocks = stocks_df.Name

selected_stock = st.selectbox('Search for Stocks', stocks)
# print(f'Selected Stock : {selected_stock} |\n')

ticker = stocks_df.loc[stocks_df['Name']==selected_stock , ['Ticker']]
ticker = str(ticker.values[0][0])
# print(f'\n\nTicker{ticker}')

stock = Data_API()
data_load_state = st.text('Loading data...')
data = stock.data_call(ticker,
					   TODAY,
					   START)
data_load_state.text('Loading data... done!')


df = stock.df
# st.subheader('Raw data')
# st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure(data=[go.Candlestick(x=df.index,
                						open=df['Open'],
                						high=df['High'],
                						low=df['Low'],
                						close=df['Close'])
						]
					)
	fig.layout.update( xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

	
plot_raw_data()

st.write(stock.describe())

scalled_data = stock.scaller(data)

data_formatter = Data_Formatter()

X , y , x_init = data_formatter.get_train(scalled_data)


bilstm_model = BiLSTM()
bilstm_model.train(X ,y)


def plot_predictions(data , forecast , preds) :
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data.index , y = data , mode ='lines' , name = "Actual Close Price" ))
	fig.add_trace(go.Scatter(x=forecast , y=preds , mode='lines' , name = "Predicted Close Prices"))
	fig.layout.update(xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

st.subheader("See future Prices of the Stock")

future_days = st.slider('Select the number of days to forecast in future', 30, 365)

forecast = pd.date_range(start =TODAY , periods= future_days ,freq='D')
preds = bilstm_model.predict_future(x_init , future_days)

preds = stock.inverse_scaller(preds)
preds = preds.reshape(-1)

print("\n Shape \n")
print(forecast.shape , preds.shape)
plot_predictions(data , forecast , preds)


