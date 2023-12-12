## ESERCITAZIONE ##
# Accediamo alla record github e prendiamo il dataset per i dati del covid della protezione civile
# Noi siamo interessati al csv: 
#   https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv
# Bisogna fare una dashboar:
#   Togliere tutte le colonne che presentavano almeno un valore mancante 
#   Si conserva solo la data senza ora
#   Come filtri Abbiamo regioni (da estrapolare direttamente da py - prendere tutti i valori unici)
#   nella barra laterare start_date - end_date
#   Fare due colonne in cui mettiamo:
#       positivi per regione 
#       ospedale per regioni
#   Fare grafici in cui ho tante line quante sono le regioni (variabile color)
#   Animazione dell'andamento del covid in varie regioni dle covid

# Library import
import streamlit as st
import pandas as pd
import numpy as np
#import plotly.express as px
#from prophet import Prophet

# Page configuration
st.set_page_config(
    page_title='Parasites DashBoard',
    page_icon=':🦗:'
    )
st.title('Environmental Dashboard: Male Parasites Under the Lens 🍃🪳')

# Data Source 
url = '../Dataset_ISBI.csv'
url_target = '../Dataset_ISBI_Target.csv'

data_url = 'time'
data_url_target = 'Date'

left_column, right_column = st.columns(2)
left_check = left_column.checkbox("Dataset without target")
right_check = right_column.checkbox("Dataset with target")
if (left_check):
    left_column.write("5")
if(right_check):
    right_column.write("42")





# Data
def _lower(string):
    return str.lower(string)

@st.cache_data
def load_data(url, data):
    df = pd.read_csv(url, parse_dates=[data]) 
    df.rename(_lower, axis="columns", inplace=True)
    df.fillna(method='ffill', inplace=True)    # Sostituisci i valori mancanti con il valore precedente nella colonna
    #df[data] = df[data].dt.date  # Cancello le ore
    return df

df = load_data(url, data_url)
df_target = load_data(url_target, data_url_target)






'''

st.sidebar.subheader('Ticker query parameters')
# Ticker sidebar
with open('ticker_symbols.txt', 'r') as fp:
    ticker_list = fp.read().split('\n')
ticker_selection = st.sidebar.selectbox(label='Stock ticker', options=ticker_list, index=ticker_list.index('AAPL'))
period_list = ['6mo', '1y', '2y', '5y', '10y', 'max']
period_selection = st.sidebar.selectbox(label='Period', options=period_list, index=period_list.index('2y'))

# Retrieving tickers data
ticker_data = yf.Ticker(ticker_selection)
ticker_df = ticker_data.history(period=period_selection)
ticker_df = ticker_df.rename_axis('Date').reset_index()
ticker_df['Date'] = ticker_df['Date'].dt.date

# Prophet sidebar
st.sidebar.subheader('Prophet parameters configuration')
horizon_selection = st.sidebar.slider('Forecasting horizon (days)', min_value=1, max_value=365, value=90)
growth_selection = st.sidebar.radio(label='Growth', options=['linear', 'logistic'])
if growth_selection == 'logistic':
    st.sidebar.info('Configure logistic growth saturation as a percentage of latest Close')
    cap = st.sidebar.slider('Constant carrying capacity', min_value=1.0, max_value=1.5, value=1.2)
    cap_close = cap*ticker_df['Close'].iloc[-1]
    ticker_df['cap']=cap_close
seasonality_selection = st.sidebar.radio(label='Seasonality', options=['additive', 'multiplicative'])
with st.sidebar.expander('Seasonality components'):
    weekly_selection = st.checkbox('Weekly')
    monthly_selection = st.checkbox('Monthly', value=True)
    yearly_selection = st.checkbox('Yearly', value=True)
with open('holiday_countries.txt', 'r') as fp:
    holiday_country_list = fp.read().split('\n')
    holiday_country_list.insert(0, 'None')
holiday_country_selection = st.sidebar.selectbox(label="Holiday country", options=holiday_country_list)

# Ticker information
company_name = ticker_data.info['longName']
st.header(company_name)
company_summary = ticker_data.info['longBusinessSummary']
st.info(company_summary)

st.header('Ticker data')
# Ticker data
var_list = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
st.dataframe(ticker_df[var_list])

st.header('Forecasting')
# Prophet model fitting
with st.spinner('Model fitting..'):
    prophet_df = ticker_df.rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(
        seasonality_mode=seasonality_selection,
        weekly_seasonality=weekly_selection,
        yearly_seasonality=yearly_selection,
        growth=growth_selection,
        )
    if holiday_country_selection != 'None':
        model.add_country_holidays(country_name=holiday_country_selection)      
    if monthly_selection:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(prophet_df)

# Prophet model forecasting
with st.spinner('Making predictions..'):
    future = model.make_future_dataframe(periods=horizon_selection, freq='D')
    if growth_selection == 'logistic':
        future['cap'] = cap_close
    forecast = model.predict(future)

# Prophet forecast plot
fig = px.scatter(prophet_df, x='ds', y='y', labels={'ds': 'Day', 'y': 'Close'})
fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat')
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='yhat_lower')
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='yhat_upper')
st.plotly_chart(fig)

'''

