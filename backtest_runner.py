from data_loader import load_local_price_data
from strategy.entry_logic import EntryLogic
from strategy.trade_simulator import TradeSimulator
from strategy.entry_diagnostics import EntryDiagnostics
import pandas as pd
from strategy.institutional_entry import InstitutionalEntrySystem


df_m15 = load_local_price_data("data/EURUSD_M15_dec24_apr25.csv")
df_h4 = load_local_price_data("data/EURUSD_H4_dec24_apr25.csv")


# entry_logic = EntryLogic(df_m15, df_h4)

# for i in range(len(df_m15)):
#     entry_logic.check_entry_signal(i)

# logs = entry_logic.get_logs()

# print("📊 Backtest-Auswertung:")
# print("Trendverteilung:", logs["trend"])
# print("Versuche mit klarer Trendlage:", logs["checked_setups"])
# print("Breakouts erkannt:", logs["breakouts"])
# print("Retests erkannt:", logs["retests"])
# print("Rejection-Kerzen erkannt:", logs["rejections"])
# print("Vollständige Entry-Signale:", logs["entries"])
# print("\n✅ Backtest abgeschlossen – Wins: 0, Losses: 0 (noch ohne Trade-Management)")

# # 3. Simulation ausführen
# sim = TradeSimulator(df_m15, entry_logic, crv=2.0)
# sim.run()
# sim.report()

# # 1. Trade-Log und Preisdaten laden
# df_m15 = load_local_price_data("data/EURUSD_M15_dec24_apr25.csv")
# df_trades = pd.read_csv("results/trade_log.csv")

# # 2. Diagnostik ausführen
# from strategy.entry_diagnostics import EntryDiagnostics
# diag = EntryDiagnostics(df_m15, df_trades)
# result_df = diag.analyze()

# # 3. Speichern
# result_df.to_csv("results/entry_diagnostics.csv", index=False)



inst = InstitutionalEntrySystem(df_m15, df_h4)
entries = inst.evaluate_entries(min_score=3)
inst.export_diagnostics()
