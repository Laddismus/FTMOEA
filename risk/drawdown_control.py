import datetime
import json

class DrawdownControl:
    def __init__(self, config_path="config/settings.json"):
        self.load_config(config_path)
        self.last_reset_day = None
        self.start_of_day_equity = self.initial_balance
        self.daily_loss_limit = None
        self.equity_floor = self.initial_balance * (1 - self.safe_max_total_loss_pct / 100)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.initial_balance = config["initial_balance"]
        self.safe_max_daily_loss_pct = config["safe_max_daily_loss_pct"]
        self.safe_max_total_loss_pct = config["safe_max_total_loss_pct"]

    def reset_daily_loss_limit(self, current_equity, today):
        if self.last_reset_day != today:
            self.start_of_day_equity = current_equity
            self.daily_loss_limit = self.start_of_day_equity - (self.initial_balance * self.safe_max_daily_loss_pct / 100)
            self.last_reset_day = today

    def is_within_limits(self, current_equity):
        today = datetime.date.today()
        self.reset_daily_loss_limit(current_equity, today)
        if current_equity < self.daily_loss_limit:
            return False
        if current_equity < self.equity_floor:
            return False
        return True
