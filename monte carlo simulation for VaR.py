import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as dr
import math
def brownian_motion(RndWalk, drift_mean, vol):
    price_increment = np.exp(drift_mean + (vol * RndWalk))
    return price_increment


df = dr.data.get_data_yahoo('ibm', start = '2017-09-27', end = '2018-10-23')
df = df[['Adj Close']]
df = df.rename(columns={"Adj Close": "AdClose"})
ret = df[['AdClose']].pct_change().dropna()
logret = sp.log(ret + 1)


paths = 1000
steps  = 5
s0 = df.AdClose[-1]
vol= float(np.std(logret))
mu = float(logret.mean())


#Browian motion assumption   d(lns) = (mu - 0.5 sigma^2)dt +sigma dX
drift_mean = mu -(0.5* vol**2)

simulation_startdate = df.index[-1]
daterng = pd.bdate_range(simulation_startdate, periods = steps + 1)
simul_paths = pd.DataFrame(index = daterng, columns = range(paths))

simul_paths.iloc[-1:].loc[simulation_startdate] = df.AdClose[-1]
RndWalk = pd.DataFrame(np.random.randn(steps + 1,paths))

for j in range(paths):
        for i in range(1, len(simul_paths)):
            simul_paths.iloc[i,j] = simul_paths.iloc[i-1,j] * brownian_motion(RndWalk.iloc[i,j], drift_mean, vol)


# Get VaR
simul_paths.iloc[-1:]
confidence = 0.05

lastrowname = simul_paths.index[-1]

finalValues = (simul_paths.iloc[-1:])

numColumns = len(finalValues.columns)
VARColumn = math.floor(numColumns * (1 - confidence))
type(finalValues)

VaR = finalValues.T.sort_values(by = ["2018-10-30T00:00:00.000000000"],ascending = True).T.iloc[0,VARColumn]
