!pip install neuralprophet
import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt

df = pd.read_csv('W.csv')
df['Date'] = pd.to_datetime(df['Date'])

plt.plot(df['Date'], df['AirTempCelsius'])
plt.show()
df['Y'] = melb['Date'].apply(lambda x: x.year)

plt.plot(df['Date'], df['AirTempCelsius'])
plt.show()
data = df[['Date', 'AirTempCelsius']] 
data.dropna(inplace=True)
data.columns = ['ds', 'y'] 
data.head()

m = NeuralProphet()
model = m.fit(data, freq='D', epochs=1000)

future = m.make_future_dataframe(data, periods=12000)
forecast = m.predict(future)
forecast.head()
plot1 = m.plot(forecast)
plt2 = m.plot_components(forecast)