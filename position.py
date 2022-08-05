from math import isnan

class Position:
  def close(self, price, step=None):
    self.pnl += self.unrealized_pnl(price)
    self.pnl = max(self.pnl, -self.balance)
    self._reset_order()
    print(f'{f"({step}) " if step is not None else ""}Closed position at {price:.2f}, pnl={self.pnl:.2f}')

  def is_open(self):
    return self.size != 0.0

  def has_blown_up(self, tolerance=0.01):
    return abs(self.pnl + self.balance) <= tolerance

  def open(self, price, size, stop_loss, take_profit, tolerance=0.01, step=None):
    size = size * (self.balance + self.pnl) / price
    if isnan(size) or abs(size) < tolerance:
      return
    self.entry_price = price
    self.size = size
    sign = 1 if size > 0 else -1
    self.stop_loss = price * (1 - sign * stop_loss)
    self.take_profit = price * (1 + sign * take_profit)
    print(f'{f"({step}) " if step is not None else ""}Opened position price={price:.2f}, size={self.size:.2f}, sl={self.stop_loss:.2f}, tp={self.take_profit:.2f}')

  def reset(self):
    self.balance = 1.0
    self.pnl = 0.0
    self._reset_order()

  def _reset_order(self):
    self.entry_price = None
    self.size = 0.0
    self.stop_loss = None
    self.take_profit = None

  def total_pnl(self, price):
    return self.pnl + self.unrealized_pnl(price)

  def unrealized_pnl(self, price):
    return 0.0 if self.size == 0.0 else self.size * (price - self.entry_price)
