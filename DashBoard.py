
# Library import
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
#import plotly.express as px
#from prophet import Prophet

# Page configuration
st.set_page_config(
    page_title='Parasites DashBoard',
    page_icon=':🦗:',
    layout='wide',
    initial_sidebar_state='expanded'  # Espandi la barra laterale inizialmente, se desiderato
    )
st.title('Environmental Dashboard: Male Parasites Under the Lens 🍃🪳')

# Data Source 
url = 'https://raw.githubusercontent.com/Dad-cip/Information-System-e-Business-Intelligent/main/Dataset_ISBI.csv'
url_target = 'https://raw.githubusercontent.com/Dad-cip/Information-System-e-Business-Intelligent/main/Dataset_ISBI_Target.csv'

data_url = 'time'
data_url_target = 'Date'

# Data
def _lower(string):
    return str.lower(string)

@st.cache_data
def load_data(url, data):
#    df = pd.read_csv(url, sep=';', parse_dates=[data], index_col=data)
    df = pd.read_csv(url, sep=';', parse_dates=[data])
    #df.rename(_lower, axis="columns", inplace=True)
    df.fillna(method='ffill', inplace=True)
    #df[data] = df.index.date     # Cancella le ore dalla colonna "Date"
    df['temperature_mean'] = df['temperature_mean'].str.replace(',', '.').astype(float)
    df['selected'] = [False] * len(df)      # Aggiunge una colonna che permette di selezionare una riga
    return df

df = load_data(url, data_url)
df_target = load_data(url_target, data_url_target)

# Inizializazione delle variabili di stato
if 'selected_df' not in st.session_state:
    st.session_state['selected_df'] = df.copy()
if 'selected_df_target' not in st.session_state:
    st.session_state['selected_df_target'] = df_target.copy()

#df.index = df.index.date
df[data_url] = df[data_url].dt.date     # Cancella le ore dalla colonna "Date"

left_column, right_column = st.columns(2)
left_check = left_column.checkbox("Dataset without target")
right_check = right_column.checkbox("Dataset with target")

# Visualizza i grafici quando vengono selezionati i checkbox
if left_check:
    left_column.subheader("Temperature and Humidity Trend")
    #left_column.line_chart(df.reset_index(), x='time', y='temperature_mean', width=1200, height=400, color='#ff0000')
    #left_column.line_chart(df[['time', 'temperature_mean', 'relativehumidity_mean']], x='time', y=['temperature_mean', 'relativehumidity_mean'])
    #left_chart=px.line(df,x='time',y=['relativehumidity_mean','temperature_mean'])
    #left_column.plotly_chart(left_chart, use_container_width=True)
    left_chart = go.Figure()
    left_chart.add_trace(go.Scatter(x=df[data_url], y=df['relativehumidity_mean'],
                    mode='lines',
                    name='humidity'))
    left_chart.add_trace(go.Scatter(x=df[data_url], y=df['temperature_mean'],
                    mode='lines',
                    name='temperature'))
    for row_index, row in st.session_state.selected_df.iterrows():
        if row['selected']:
            selected_sample = df.iloc[row_index]
            left_chart.add_trace(go.Scatter(x=[selected_sample[data_url]], y=[selected_sample['temperature_mean']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
            left_chart.add_trace(go.Scatter(x=[selected_sample[data_url]], y=[selected_sample['relativehumidity_mean']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
    left_column.plotly_chart(left_chart, use_container_width=True)

if right_check:
    right_column.subheader("Temperature and Humidity Trend")
    #right_column.line_chart(df_target.reset_index(), x='Date', y='temperature_mean', width=1200, height=400, color='#ff0000')
    #right_chart=px.line(df_target,x='Date',y=['relativehumidity_mean','temperature_mean'])
    #right_column.plotly_chart(right_chart)
    filtered_df = df_target[df_target['no. of Adult males'] != 0]
    right_chart = go.Figure()
    right_chart.add_trace(go.Scatter(x=df_target[data_url_target], y=df_target['relativehumidity_mean'],
                    mode='lines',
                    name='humidity'))
    right_chart.add_trace(go.Scatter(x=df_target[data_url_target], y=df_target['temperature_mean'],
                    mode='lines',
                    name='temperature'))
    right_chart.add_trace(go.Scatter(x=filtered_df[data_url_target], y=filtered_df['no. of Adult males'],
                    mode='markers',
                    name='no. parasites',
                    marker=dict(
                        size=filtered_df['no. of Adult males'],  # Marker size based on the column values
                        sizemode='area',  # Options: 'diameter', 'area'
                        sizeref=0.1,  # Adjust the size reference as needed
                    ),
                    ))
    for row_index, row in st.session_state.selected_df_target.iterrows():
        if row['selected']:
            selected_sample_right = df_target.iloc[row_index]
            right_chart.add_trace(go.Scatter(x=[selected_sample_right[data_url_target]], y=[selected_sample_right['temperature_mean']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
            right_chart.add_trace(go.Scatter(x=[selected_sample_right[data_url_target]], y=[selected_sample_right['relativehumidity_mean']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
            if selected_sample_right['no. of Adult males'] > 0:
                right_chart.add_trace(go.Scatter(x=[selected_sample_right[data_url_target]], y=[selected_sample_right['no. of Adult males']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
    right_column.plotly_chart(right_chart, use_container_width=True)

# Definizione della callback di aggiornamento della variabile di stato
def update(df,key):
    for elem in st.session_state[key]['edited_rows']:
        st.session_state[df]['selected'][elem] = st.session_state[key]['edited_rows'][elem]['selected']

left_column.subheader("Dataframe: Dataset without target")
left_column.data_editor(df, key="left_editor", 
                        disabled=['time', 'temperature_mean', 'relativehumidity_mean'], 
                        hide_index=True, on_change=update, args=('selected_df','left_editor'))
right_column.subheader("Dataframe: Dataset with target")
right_column.data_editor(df_target, key="right_editor", 
                        disabled=['Date', 'no. of Adult males', 'temperature_mean', 'relativehumidity_mean'], 
                        hide_index=True, on_change=update, args=('selected_df_target','right_editor'))







# Studiamo la CORRELAZIONE dei dati 
left_column.markdown("<h1 style='text-align: center;'>Correlazione <span style='font-size: 28px;'>without target</span></h1>", unsafe_allow_html=True)

df_no_date = df.drop(columns=data_url)

left_left_column, left_right_column = left_column.columns(2)

selected_x = left_left_column.selectbox("Seleziona colonna per l'asse x", df_no_date.columns)
selected_y = left_right_column.selectbox("Seleziona colonna per l'asse y", df_no_date.columns)

corr = df[selected_x].corr(df[selected_y])
# Crea un grafico di dispersione con seaborn
correlazione = plt.figure(figsize=(8, 6))
sns.regplot(x=selected_x, y=selected_y, data=df, scatter_kws={'s': 100})
plt.xlabel(selected_x)
plt.ylabel(selected_y)
plt.text(df[selected_x].min(), df[selected_y].max(), f'Correlation: {corr:.2f}', ha='left', va='bottom')
plt.grid(True)
plt.show()
left_column.pyplot(correlazione)



corr_target = df_target['temperature_mean'].corr(df_target['relativehumidity_mean'])
# Crea un grafico di dispersione con seaborn
'''
correlazione = plt.figure(figsize=(8, 6))
sns.regplot(x='temperature_mean', y='relativehumidity_mean', data=df, scatter_kws={'s': 100})
plt.title('Scatter Plot tra Temperatura Media e Umidità Relativa Media')
plt.xlabel('Temperatura Media')
plt.ylabel('Umidità Relativa Media')
plt.text(df['temperature_mean'].min(), df['relativehumidity_mean'].max(), f'Correlation: {correlation:.2f}', ha='left', va='bottom')
plt.grid(True)
plt.show()
'''



right_column.pyplot(correlazione)




























## SIDEBAR ##      
st.sidebar.title('Dataset without Target')  
st.sidebar.subheader('Temporal Filter')

# Definizione di variabili di stato per il filtro temporale
start_date = st.sidebar.date_input("Start Date", df[data_url].min(), df[data_url].min(), df[data_url].max())
end_date = st.sidebar.date_input("End Date", df[data_url].max(), df[data_url].min(), df[data_url].max())

# Filtrare i dati in base alle date selezionate
filtered_df = df[(df[data_url] >= start_date) & (df[data_url] <= end_date)]

# Visualizzazione del grafico basato sulle date selezionate
st.subheader('Temperature and Humidity Trend')
filtered_chart = go.Figure()

filtered_chart.add_trace(go.Scatter(x=filtered_df[data_url], y=filtered_df['relativehumidity_mean'],
                    mode='lines',
                    name='humidity'))
filtered_chart.add_trace(go.Scatter(x=filtered_df[data_url], y=filtered_df['temperature_mean'],
                    mode='lines',
                    name='temperature'))

st.plotly_chart(filtered_chart, use_container_width=True)
#left_column.plotly_chart(left_chart, use_container_width=True)














st.sidebar.title('Dataset with Target')  
st.sidebar.subheader('Temporal Filter (df_target)')

'''
data_string = "15-giu"

# Mappa dei nomi dei mesi abbreviati ai corrispondenti numeri del mese
mesi_italiani = {
    'gen': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'mag': 5, 'giu': 6, 'lug': 7, 'ago': 8,
    'set': 9, 'ott': 10, 'nov': 11, 'dic': 12
}

# Esempio di conversione
giorno, mese_abbreviato = data_string.split('-')
mese = mesi_italiani.get(mese_abbreviato.lower())
anno = 2023  # Assumiamo un anno fisso, puoi regolarlo come necessario

# Converte la stringa in un oggetto datetime
data = datetime(anno, mese, int(giorno))







# Definizione di variabili di stato per il filtro temporale per df_target
df_target['Date'] = pd.to_datetime(df_target[data_url_target])
start_date_df_target = st.sidebar.date_input("Start Date (df_target)", df_target[data_url_target].min(), df_target[data_url_target].min(), df_target[data_url_target].max())
end_date_df_target = st.sidebar.date_input("End Date (df_target)", df_target[data_url_target].max(), df_target[data_url_target].min(), df_target[data_url_target].max())

# Conversione delle date per df_target
start_date_df_target = start_date_df_target.strftime('%d-%b')
end_date_df_target = end_date_df_target.strftime('%d-%b')

# Filtrare i dati in base alle date selezionate per df_target
filtered_df_target = df_target[(df_target['date'] >= start_date_df_target) & (df_target['date'] <= end_date_df_target)]

# Visualizzazione del grafico basato sulle date selezionate per df_target
st.subheader('Temperature and Humidity Trend (df_target)')
filtered_chart_df_target = go.Figure()

filtered_chart_df_target.add_trace(go.Scatter(x=filtered_df_target['date'], y=filtered_df_target['relativehumidity_mean'],
                    mode='lines',
                    name='humidity'))
filtered_chart_df_target.add_trace(go.Scatter(x=filtered_df_target['date'], y=filtered_df_target['temperature_mean'],
                    mode='lines',
                    name='temperature'))

# Aggiungi qui eventuali ulteriori tracce o personalizzazioni del grafico per df_target

st.plotly_chart(filtered_chart_df_target, use_container_width=True)


'''








'''
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

