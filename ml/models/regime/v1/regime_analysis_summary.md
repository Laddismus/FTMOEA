# Regime Analysis Summary

- File: `F:\TradingBot\FTMOEA\data_pipeline\data\features\EURUSD_5m_BACKTEST.parquet`  
- Rows analyzed: 20000  
- Mean conf: 0.658  
- Drift rate: 0.0  

## Regime counts

regime
Range_Quiet       5774
Range_Volatile    5566
Bear_Volatile     2708
Bull_Volatile     2666
Bull_Quiet        1744
Bear_Quiet        1542

## Mean ATR by regime (pips)

regime
Bear_Quiet        3.45
Bear_Volatile     7.51
Bull_Quiet        3.46
Bull_Volatile     7.38
Range_Quiet       3.56
Range_Volatile    6.96

## Transition matrix

regime          Bear_Quiet  Bear_Volatile  Bull_Quiet  Bull_Volatile  Range_Quiet  Range_Volatile
regime                                                                                           
Bear_Quiet           0.797          0.034       0.056          0.001        0.110           0.001
Bear_Volatile        0.017          0.839       0.003          0.053        0.004           0.084
Bull_Quiet           0.055          0.002       0.822          0.036        0.084           0.002
Bull_Volatile        0.001          0.066       0.014          0.846        0.008           0.066
Range_Quiet          0.028          0.002       0.029          0.004        0.906           0.031
Range_Volatile       0.001          0.034       0.001          0.033        0.035           0.895
