from ib_insync import *
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

contract = Forex('EURUSD')
bars = ib.reqHistoricalData(contract,
                            endDateTime='',
                            durationStr='30 D',
                            barSizeSetting='1 hour',
                            whatToShow='MIDPOINT',
                            useRTH=True)

# convert to pandas dataframe:
df = util.df(bars)
print(df[['date', 'open', 'high', 'low', 'close']])
ib.disconnect()
