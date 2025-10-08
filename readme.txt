1. data_pipeline: 

2. ml/regime/train:
python -u F:\TradingBot\FTMOEA\ml\regime\train_regime.py "F:\TradingBot\FTMOEA\ml\regime\configs\regime.yaml"

3. ml/regime:

PS F:\TradingBot\FTMOEA\ml\regime> python -u "F:\TradingBot\FTMOEA\ml\regime\Regime_analysis.py" `
>>   --artifacts "F:\TradingBot\FTMOEA\ml\models\regime\v1" `
>>   --features "F:\TradingBot\FTMOEA\data_pipeline\data\features\EURUSD_5m_BACKTEST.parquet" `   
>>   --asset EURUSD `
>>   --tf 5m `
>>   --rows 20000 `
>>   --export_csv "F:\TradingBot\FTMOEA\ml\regime\regime_monitor_EURUSD_5m.csv"
>>

