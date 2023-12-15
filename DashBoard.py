
# Library import
from libraries import *

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
    df['temperature_mean'] = df['temperature_mean'].str.replace(',', '.').astype(float)
    df['selected'] = [False] * len(df)      # Aggiunge una colonna che permette di selezionare una riga
    return df

df = load_data(url, data_url)
df_target = load_data(url_target, data_url_target)

# Aggiustiamo la colonna relativa alla data del dataframe senza target
df[data_url] = df[data_url].dt.date     # Cancella le ore dalla colonna "Date"

# Aggiustiamo la colonna relativa alla data del dataframe con target
mappa_mesi = {
    'gen': 'Jan-2022',
    'feb': 'Feb-2022',
    'mar': 'Mar-2022',
    'apr': 'Apr-2022',
    'mag': 'May-2022',
    'giu': 'Jun-2022',
    'lug': 'Jul-2022',
    'ago': 'Aug-2022',
    'set': 'Sep-2022',
    'ott': 'Oct-2022',
    'nov': 'Nov-2022',
    'dic': 'Dec-2022'
}
for ita, eng in mappa_mesi.items():
    for i in range(len(df_target[data_url_target])):
        new_value = df_target.iloc[i, df_target.columns.get_loc(data_url_target)].replace(ita, eng)
        df_target.iloc[i, df_target.columns.get_loc(data_url_target)] = new_value
df_target[data_url_target] = [datetime.strptime(data_string, '%d-%b-%Y') for data_string in df_target[data_url_target]]
df_target[data_url_target] = df_target[data_url_target].dt.date


# Inizializazione delle variabili di stato
if 'selected_df' not in st.session_state:
    st.session_state['selected_df'] = df.copy()
if 'selected_df_target' not in st.session_state:
    st.session_state['selected_df_target'] = df_target.copy()
if 'temp_filtered_df' not in st.session_state:
    st.session_state['temp_filtered_df'] = df.copy()
if 'temp_filtered_df_target' not in st.session_state:
    st.session_state['temp_filtered_df_target'] = df_target.copy()


## SIDEBAR ##      
st.sidebar.title('Temporal Filter')  
st.sidebar.subheader('Dataset without Target')

# Definizione di variabili di stato per il filtro temporale
side_left_column, side_right_column = st.sidebar.columns(2)

start_date = side_left_column.date_input("Start Date", df[data_url].min(), df[data_url].min(), df[data_url].max())
end_date = side_right_column.date_input("End Date", df[data_url].max(), df[data_url].min(), df[data_url].max())

# Filtrare i dati in base alle date selezionate
temp_filtered_df = st.session_state.selected_df[(df[data_url] >= start_date) & (df[data_url] <= end_date)]
st.session_state.temp_filtered_df = temp_filtered_df


st.sidebar.subheader('Dataset with target')

# Definizione di variabili di stato per il filtro temporale
side_left_column, side_right_column = st.sidebar.columns(2)

start_date_target = side_left_column.date_input("Start Date", df_target[data_url_target].min(), df_target[data_url_target].min(), df_target[data_url_target].max())
end_date_target = side_right_column.date_input("End Date", df_target[data_url_target].max(), df_target[data_url_target].min(), df_target[data_url_target].max())

# Filtrare i dati in base alle date selezionate
temp_filtered_df_target = st.session_state.selected_df_target[(df_target[data_url_target] >= start_date_target) & (df_target[data_url_target] <= end_date_target)]
st.session_state.temp_filtered_df_target = temp_filtered_df_target

left_column, right_column = st.columns(2)
left_check = left_column.checkbox("Dataset without target")
right_check = right_column.checkbox("Dataset with target")

# Visualizza i grafici quando vengono selezionati i checkbox
if left_check:
    left_column.subheader("Temperature and Humidity Trend")
    left_chart = go.Figure()
    left_chart.add_trace(go.Scatter(x=temp_filtered_df[data_url], y=temp_filtered_df['relativehumidity_mean'],
                    mode='lines',
                    name='humidity'))
    left_chart.add_trace(go.Scatter(x=temp_filtered_df[data_url], y=temp_filtered_df['temperature_mean'],
                    mode='lines',
                    name='temperature'))
    for row_index, row in st.session_state.selected_df.iterrows():
        if row['selected'] and (row_index in st.session_state.temp_filtered_df.index):
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
    no_zero_df = temp_filtered_df_target[temp_filtered_df_target['no. of Adult males'] != 0]
    right_chart = go.Figure()
    right_chart.add_trace(go.Scatter(x=temp_filtered_df_target[data_url_target], y=temp_filtered_df_target['relativehumidity_mean'],
                    mode='lines',
                    name='humidity'))
    right_chart.add_trace(go.Scatter(x=temp_filtered_df_target[data_url_target], y=temp_filtered_df_target['temperature_mean'],
                    mode='lines',
                    name='temperature'))
    right_chart.add_trace(go.Scatter(x=no_zero_df[data_url_target], y=no_zero_df['no. of Adult males'],
                    mode='markers',
                    name='no. parasites',
                    marker=dict(
                        size=no_zero_df['no. of Adult males'],  # Marker size based on the column values
                        sizemode='area',  # Options: 'diameter', 'area'
                        sizeref=0.1,  # Adjust the size reference as needed
                    ),
                    ))
    for row_index, row in st.session_state.selected_df_target.iterrows():
        if row['selected'] and (row_index in st.session_state.temp_filtered_df_target.index):
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
                        column_order=('selected', 'time', 'temperature_mean', 'relativehumidity_mean'),
                        disabled=['time', 'temperature_mean', 'relativehumidity_mean'], 
                        hide_index=True, on_change=update, args=('selected_df','left_editor'))
right_column.subheader("Dataframe: Dataset with target")
right_column.data_editor(df_target, key="right_editor", 
                         column_order=('selected', 'Date', 'no. of Adult males', 'temperature_mean', 'relativehumidity_mean'),
                         disabled=['Date', 'no. of Adult males', 'temperature_mean', 'relativehumidity_mean'], 
                         hide_index=True, on_change=update, args=('selected_df_target','right_editor'))


# Studiamo la CORRELAZIONE dei dati 
left_column.markdown("<h1>Correlazione <span style='font-size: 28px;'>without target</span></h1>", unsafe_allow_html=True)

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
#plt.show()
left_column.pyplot(correlazione)


right_column.markdown("<h1>Correlazione <span style='font-size: 28px;'>with target</span></h1>", unsafe_allow_html=True)

df_target_no_date = df_target.drop(columns=data_url_target)

right_left_column, right_right_column = right_column.columns(2)

selected_x = right_left_column.selectbox("Seleziona colonna per l'asse x", df_target_no_date.columns)
selected_y = right_right_column.selectbox("Seleziona colonna per l'asse y", df_target_no_date.columns)

corr_target = df_target[selected_x].corr(df_target[selected_y])
# Crea un grafico di dispersione con seaborn
correlazione_target = plt.figure(figsize=(8, 6))
sns.regplot(x=selected_x, y=selected_y, data=df_target, scatter_kws={'s': 100})
plt.xlabel(selected_x)
plt.ylabel(selected_y)
plt.text(df_target[selected_x].min(), df_target[selected_y].max(), f'Correlation: {corr_target:.2f}', ha='left', va='bottom')
plt.grid(True)
right_column.pyplot(correlazione_target)


st.markdown("<h1 style='text-align: center;'>INSPECTION DATASET TARGET</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Autocorrelation</h2>", unsafe_allow_html=True)

left_column, center_column, right_column = st.columns(3)

# Slider LAGS
lags = center_column.slider(label="lags", min_value=1, max_value=52, step=1, value=20)

left_column, right_column = st.columns(2)

# Grafico ACF
fig_acf = sgt.plot_acf(df_target['no. of Adult males'], lags=lags, zero=False, title="Autocorrelation of no. of Adult males")
left_column.pyplot(fig_acf)

# Grafico PACF
fig_pacf = sgt.plot_pacf(df_target['no. of Adult males'], lags=lags, zero=False, method='ols', title="PACF of no. of Adult males")
right_column.pyplot(fig_pacf)


df_target['Date'] = pd.to_datetime(df_target['Date'], format='%d-%b')
# Set year for the second dataframe as 2022
df_target['Date'] = df_target['Date'].apply(lambda x: x.replace(year=2022))
df_target.set_index('Date', inplace=True)
df_target = df_target.drop(columns='selected')

st.markdown("<h2 style='text-align: center;'>Differentiation</h2>", unsafe_allow_html=True)

left_column, right_column = st.columns(2)
left_column.text('Select the order of differentiation:')
left_column.text('\n\n')
diff_order = right_column.number_input("", min_value=1, max_value=10, value=1, step=1, label_visibility='collapsed')
df_diff = df_target.diff(diff_order).dropna()
    
fig, ax = plt.subplots(figsize=(12, 6)) 
ax.plot(df_target.index, df_target['no. of Adult males']) 
ax.set_title('Before Differentiation') 
ax.set_xlabel('Date') 
ax.set_ylabel('no. of Adult males') 
ax.legend() 
left_column.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 6)) 
ax.plot(df_diff.index, df_diff['no. of Adult males']) 
ax.set_title('After Differentiation') 
ax.set_xlabel('Date') 
ax.set_ylabel('no. of Adult males') 
ax.legend() 
right_column.pyplot(fig)




# SIDEBAR MODELS
st.sidebar.header('Model List')

with st.sidebar:
    var_check = st.checkbox("VAR")
    arimax_check = st.checkbox("ARIMAX")
    rt_check = st.checkbox("Regression Tree")
    nn_check = st.checkbox("Neural Network")

    # Make splitting 80-10-10
    train_size = int(len(df_diff)*0.8)
    val_size = int(len(df_diff)*0.1)
    df_diff_train = df_diff.iloc[:train_size+1]
    df_diff_val = df_diff.iloc[train_size+1:train_size+val_size+1]
    df_diff_test = df_diff.iloc[train_size+val_size+1:]

@st.cache_data
def plot_differencies(_x1,y1,_x2,y2):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(_x1, y1, label='Actual', color='blue')
    plt.plot(_x2, y2, label='Predicted', color='red')
    plt.xlabel('Data')
    plt.ylabel('No. of Adult males')
    plt.title('Confronto tra Valori Effettivi e Previsti')
    plt.legend()
    return fig


if var_check:
    st.markdown("<h2>VAR Model</h2>", unsafe_allow_html=True)
    #VAR
    #lags_var = st.form()
    # Aggiungi una descrizione sopra la casella di input numerica

    left_column, right_column = st.columns(2)
    left_column.text("Enter the number of lags you want to fit:")
    
    # Chiedi all'utente di inserire un numero
    lags_var = left_column.number_input("", min_value=0, max_value=52, value=6, step=1, label_visibility='collapsed')
    
    # Let's differentiate the series in order to make it stationary

    # Create and fit model VAR on the new stationary series

    model = VAR(df_diff_train)
    results = model.fit(lags_var)

    #st.text(results.summary())

    results_fitted = results.fittedvalues.iloc[1:]

    # Compute RMSE (Root Mean Squared Error) e AIC (Akaike Information Criterion)
    min_length = min(len(df_diff_train), len(results_fitted))  # make shapes compatible
    rmse_value = rmse_function(df_diff_train.iloc[:min_length], results_fitted.iloc[:min_length])
    aic_value = aic(results.aic, len(df_diff_train.columns), df_modelwc=len(results.params))
    left_column.markdown(f'**RMSE**: {rmse_value}')
    left_column.markdown(f'**AIC**: {aic_value}')

    # Make predictions
    lag_order = results.k_ar
    forecast = results.forecast(df_diff_train.values[-lag_order:], steps=len(df_diff_test))
    target_forecast = []
    for i in range(len(df_diff_test)):
        target_forecast.append(forecast[i,0])
    
    # Confronta le previsioni con i valori effettivi del test
    comparison = pd.DataFrame({'Predicted': target_forecast, 'Actual': df_diff_test['no. of Adult males']})
    
    # Creazione di un grafico a dispersione per visualizzare i valori predetti rispetto a quelli effettivi

    # Extract only the first column from the forecast
    forecast_first_col = forecast[:, 0]
    
    # Plot actual vs predicted values for the first column
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(df_diff_train.index, df_diff_train.iloc[:, 0], label='Training', marker='o')
    ax.plot(df_diff_val.index, df_diff_val.iloc[:, 0], label='Validation', marker='s')
    ax.plot(df_diff_test.index, forecast_first_col, label='Predicted', linestyle='dashed', marker='o')
    ax.set_title('Actual vs Predicted Values (First Column)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Your Y-axis label for the first column')
    ax.legend()
    st.pyplot(fig)

    fig = plot_differencies(comparison.index, comparison['Actual'], comparison.index, comparison['Predicted'])
    right_column.pyplot(fig)

    # Print predictions
    #st.text(f'Predictions for next {df_diff_test} observations:\n{forecast}')
    
    
if arimax_check:
    st.markdown("<h2>ARIMAX Model</h2>", unsafe_allow_html=True)
    
    left_column, right_column = st.columns(2)
    left_column.text("Enter the order:")
    
    ll, cl, rl = left_column.columns(3)
    AR_ord = ll.number_input("", min_value=0, max_value=10, value=1, step=1, label_visibility='collapsed', key='lr')
    I_ord = cl.number_input("", min_value=0, max_value=10, value=1, step=1, label_visibility='collapsed', key='cr')
    MA_ord = rl.number_input("", min_value=0, max_value=10, value=1, step=1, label_visibility='collapsed', key='rr')
    
    select_order = (AR_ord,I_ord,MA_ord)
    # Concatenate training e validation
    df_diff_train_val = pd.concat([df_diff_train, df_diff_val])
    # Including multiple exogenous variables
    exog_vars = df_diff_train_val[['relativehumidity_mean','temperature_mean']]

    model = ARIMA(df_diff_train_val['no. of Adult males'], exog = exog_vars, order=select_order)
    results = model.fit()
    
    start_date = df_diff_test.index[0]
    end_date = df_diff_test.index[-1]
    forecast = results.predict(start = start_date, end = end_date, exog=df_diff_test[['relativehumidity_mean','temperature_mean']])

    # Confronta le previsioni con i valori effettivi del test
    comparison = pd.DataFrame({'Predicted': forecast, 'Actual': df_diff_test['no. of Adult males']})

    # Ora puoi analizzare la differenza tra i valori previsti e quelli effettivi
    comparison['Difference'] = comparison['Actual'] - comparison['Predicted']

    rmse = np.sqrt(mean_squared_error(comparison['Actual'], comparison['Predicted']))
    mae = mean_absolute_error(comparison['Actual'], comparison['Predicted'])

    left_column.markdown(f'**RMSE**: {rmse}')
    left_column.markdown(f'**MAE**: {mae}')
    
    fig = plot_differencies(comparison.index, comparison['Actual'], comparison.index, comparison['Predicted'])
    right_column.pyplot(fig)
    
# Determine the number of optimal lags for each column
@st.cache_data
def find_optimal_lags(dataframe, _columnnames, max_lags=20):
    """
    Function to find the optimal number of lags for each column to satisfy the ADF test.

    Args:
    dataframe (pd.DataFrame): The dataframe containing the time series data.
    columnnames (list): List of column names to evaluate.
    max_lags (int): Maximum number of lags to test for stationarity.

    Returns:
    dict: Dictionary of column names and their respective optimal number of lags.
    """

    optimal_lags = {}
    for column in _columnnames:
        for lag in range(1, max_lags + 1):
            # Apply differencing based on the current lag
            differenced_series = dataframe[column].diff(lag).dropna()

            # Perform ADF test
            p_value = adfuller(differenced_series)[1]

            # Check if the series is stationary
            if p_value <= 0.05:
                optimal_lags[column] = lag
                break
        else:
            # If none of the lags up to max_lags make the series stationary,
            # use the max_lags value
            optimal_lags[column] = max_lags

    return optimal_lags

# Example usage:
# Assuming 'df' is your DataFrame and you want to check the first three columns
column_names = df_target.columns
optimal_lags = find_optimal_lags(df_target, column_names)

# Create a new dataframe with the lagged version of columns
@st.cache_data
def create_combined_dataset(dataframe, optimal_lags, target_column):
    """
    Create a combined dataset with original features, their lagged versions, and the original target variable.

    Args:
    dataframe (pd.DataFrame): The original dataframe.
    optimal_lags (dict): Dictionary of column names and their respective optimal number of lags.
    target_column (str): Name of the target column.

    Returns:
    pd.DataFrame: Dataframe with original features, lagged features, and the original target variable.
    """
    combined_df = pd.DataFrame(index=dataframe.index)

    # Include original features
    for column in dataframe.columns:
        if column != target_column:  # Exclude target column from lagging
            combined_df[column] = dataframe[column]

            # Create lagged features for each column based on optimal lags
            for lag in range(1, optimal_lags.get(column, 1) + 1):
                combined_df[f'{column}_lag_{lag}'] = dataframe[column].shift(lag)

    # Add the original target variable
    combined_df[target_column] = dataframe[target_column]

    # Remove rows with NaN values created by lagging
    combined_df.dropna(inplace=True)
    return combined_df

target_column = 'no. of Adult males'
combined_df = create_combined_dataset(df_target, optimal_lags, target_column)
      
if rt_check:
    st.markdown("<h2>Regression Tree Model</h2>", unsafe_allow_html=True)
    left_column, right_column = st.columns(2)
    left_column.text("Enter max_depth, min_samples_split, min_samples_leaf:")
    
    ll, cl, rl = left_column.columns(3)
    max_depth = ll.number_input("", min_value=1, max_value=10, value=1, step=1, label_visibility='collapsed', key='lr')
    min_samples_split = cl.number_input("", min_value=2, max_value=10, value=2, step=1, label_visibility='collapsed', key='cr')
    min_samples_leaf = rl.number_input("", min_value=1, max_value=10, value=1, step=1, label_visibility='collapsed', key='rr')
    
    # Dividi i dati in set di addestramento e test
    # Manual Split: 90-10
    train_size = int(len(combined_df)*0.9)
    df_train = combined_df.iloc[:train_size]
    df_test = combined_df.iloc[train_size:]
    X_train = df_train.drop(target_column, axis=1)
    y_train = df_train[target_column]
    X_test = df_test.drop(target_column, axis=1)
    y_test = df_test[target_column]
    
    rt_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    rt_model.fit(X_train, y_train)
    y_pred = rt_model.predict(X_test)

    # Valuta le prestazioni del modello
    mse = mean_squared_error(y_test, y_pred)
    left_column.markdown(f'**Mean Squared Error**: {mse}')
    
    fig = plot_differencies(df_test.index, y_test, df_test.index, y_pred)
    right_column.pyplot(fig)
    
@st.cache_data    
def preprocess_data(dataframe, target_column):
    """
    Preprocess the data for MLP training. Splits data, normalizes features, and creates an MLP model.

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    target_column (str): The name of the target column in the dataframe.

    Returns:
    tuple: (X_train, X_test, y_train, y_test, model) where `model` is an instance of a Keras Sequential model.
    """
    # Splitting the dataset into features and target
    X = dataframe.drop(target_column, axis=1)
    y = dataframe[target_column]

    # Manual split
    train_size = int(len(dataframe)*0.8)
    val_size = int(len(dataframe)*0.1)
    X_train = X.iloc[:train_size+1]
    y_train = y.iloc[:train_size+1]
    X_val = X.iloc[train_size+1:train_size+1+val_size]
    y_val = y.iloc[train_size+1:train_size+1+val_size]
    X_test = X.iloc[train_size+1+val_size:]
    y_test = y.iloc[train_size+1+val_size:]

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define MLP model
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, model

if nn_check:
    st.markdown("<h2>Neural Network Model</h2>", unsafe_allow_html=True)
    
    x_train, x_val, x_test, y_train, y_val, y_test, model = preprocess_data(combined_df, target_column)
    
    left_column, right_column = st.columns(2)
    left_column.text("Enter number of epochs, batch size:")
    
    ll, rl = left_column.columns(2)
    epochs = ll.number_input("", min_value=1, max_value=100, value=1, step=1, label_visibility='collapsed', key='lr')
    batch_size = rl.number_input("", min_value=1, max_value=len(x_train), value=1, step=1, label_visibility='collapsed', key='rr')
    
    # Fit the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)

    fig = plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    st.pyplot(fig)
    
    y_pred = model.predict(x_test)

    # Valuta le prestazioni del modello
    mse = mean_squared_error(y_test, y_pred)
    left_column.markdown(f'**Mean Squared Error**: {mse}')
    
    fig = plot_differencies(y_test.index, y_test, y_test.index, y_pred)
    right_column.pyplot(fig)
    









