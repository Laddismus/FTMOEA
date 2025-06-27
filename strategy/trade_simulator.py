import pandas as pd
import matplotlib.pyplot as plt
import os

class TradeSimulator:
    def __init__(self, df_m15, entry_logic, crv=2.0):
        self.df = df_m15
        self.entry_logic = entry_logic
        self.crv = crv
        self.initial_equity = 10000
        self.equity = self.initial_equity
        self.equity_curve = []
        self.trades = []

    def run(self):
        open_trade = None

        for i in range(len(self.df) - 1):
            if self.entry_logic.check_entry_signal(i):
                candle = self.df.iloc[i]
                idx = self.df.index[i]

                df_slice = self.df.iloc[i - 30:i] if i >= 30 else self.df.iloc[:i]
                swing_high, swing_low = self.entry_logic.find_recent_swing_high_low(df_slice)

                # Feature-basierte Filter anwenden (Momentum + Swing-Abstand)
                momentum_3 = self.df['close'].iloc[i] - self.df['close'].iloc[i - 3] if i >= 3 else 0
                dist_to_swing_low = abs(candle['close'] - swing_low) / 0.0001 if swing_low else 0  # in Pips

                if momentum_3 <= 0:
                    continue  # kein positiver Impuls → Trade auslassen
                if dist_to_swing_low <= 1:
                    continue  # zu nahe am Swing → mögliche Reversalzone

                if self.entry_logic.logs['trend']['bullish'] > self.entry_logic.logs['trend']['bearish'] and swing_low:
                    sl = swing_low
                    tp = candle['close'] + self.crv * (candle['close'] - sl)
                    open_trade = {
                        'entry': candle['close'],
                        'sl': sl,
                        'tp': tp,
                        'direction': 'long',
                        'entry_time': idx
                    }
                elif swing_high:
                    sl = swing_high
                    tp = candle['close'] - self.crv * (sl - candle['close'])
                    open_trade = {
                        'entry': candle['close'],
                        'sl': sl,
                        'tp': tp,
                        'direction': 'short',
                        'entry_time': idx
                    }

            if open_trade:
                next_candle = self.df.iloc[i + 1]
                idx = self.df.index[i + 1]

                if open_trade['direction'] == 'long':
                    if next_candle['low'] <= open_trade['sl']:
                        pnl = open_trade['sl'] - open_trade['entry']
                        self.equity += pnl
                        self.trades.append(('loss', pnl, idx, open_trade['entry_time'], open_trade['entry'], open_trade['sl'], open_trade['tp']))
                        open_trade = None
                    elif next_candle['high'] >= open_trade['tp']:
                        pnl = open_trade['tp'] - open_trade['entry']
                        self.equity += pnl
                        self.trades.append(('win', pnl, idx, open_trade['entry_time'], open_trade['entry'], open_trade['sl'], open_trade['tp']))
                        open_trade = None

                elif open_trade['direction'] == 'short':
                    if next_candle['high'] >= open_trade['sl']:
                        pnl = open_trade['entry'] - open_trade['sl']
                        self.equity += pnl
                        self.trades.append(('loss', pnl, idx, open_trade['entry_time'], open_trade['entry'], open_trade['sl'], open_trade['tp']))
                        open_trade = None
                    elif next_candle['low'] <= open_trade['tp']:
                        pnl = open_trade['entry'] - open_trade['tp']
                        self.equity += pnl
                        self.trades.append(('win', pnl, idx, open_trade['entry_time'], open_trade['entry'], open_trade['sl'], open_trade['tp']))
                        open_trade = None

            self.equity_curve.append(self.equity)

    def report(self):
        wins = sum(1 for t in self.trades if t[0] == 'win')
        losses = sum(1 for t in self.trades if t[0] == 'loss')
        print(f"✅ Trades insgesamt: {len(self.trades)}")
        print(f"🏆 Gewinne: {wins}, ❌ Verluste: {losses}")
        if wins + losses > 0:
            print(f"🎯 Trefferquote: {wins / (wins + losses):.2%}")
        print(f"📈 Finaler Kontostand: {self.equity:.2f}")

        df_trades = pd.DataFrame(self.trades, columns=['result', 'pnl', 'exit_time', 'entry_time', 'entry_price', 'sl', 'tp'])
        print("\n📋 Trade-Log (erste 10 Einträge):")
        print(df_trades.head(10))

        os.makedirs("results", exist_ok=True)
        df_trades.to_csv("results/trade_log.csv", index=False)

        plt.plot(self.equity_curve)
        plt.title("Equity Curve")
        plt.xlabel("Zeit / Trades")
        plt.ylabel("Equity ($)")
        plt.grid()
        plt.show()
