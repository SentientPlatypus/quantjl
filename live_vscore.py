import os
import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np

from alpaca.data.live.stock import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# ---------------------------------------------------------
# 1. VScore helper (replace this with YOUR real vscore)
# ---------------------------------------------------------

def compute_vscore(prices, OBS: int = 60, EPOCH: int = 1000, EXT: int = 20) -> float:
    """
    Python port of your original Julia vscore:

    Julia reference:

        function vscore(raw::Vector{Float64}, OBS::Int=60, EPOCH::Int=1000, EXT::Int=20)
            v = Float64[]
            for t in OBS:length(raw)-1
                temp = raw[t+1-OBS : t+1]
                ret = returns(temp)
                s0 = temp[end]
                μ, σ = mean(ret), std(ret)
                drift = μ + 0.5 * σ^2

                noise = cumsum(randn(EPOCH, EXT), dims=2)
                paths = s0 .* exp.(σ .* noise .+ drift .* (1:EXT)')

                sum_exceed = count(>(s0), paths)
                push!(v, sum_exceed / (EPOCH * EXT))
            end
            return (v .- mean(v)) ./ std(v)
        end

    Here we:
      - work in 0-based indexing
      - compute v for all possible t windows
      - return ONLY the last standardized vscore (for "current" time)
    """

    raw = np.asarray(prices, dtype=float)
    n = raw.size

    # Need at least OBS+1 prices to form one window (like Julia loop)
    if n < OBS + 1:
        return np.nan

    vscores = []

    # Julia: for t in OBS:length(raw)-1
    #         temp = raw[t+1-OBS : t+1]
    # 0-based Python: t runs from OBS .. n-1
    #                  temp = raw[t-OBS : t+1]
    for t in range(OBS, n):
        temp = raw[t - OBS : t + 1]   # length OBS+1

        # percent returns: (temp[i] - temp[i-1]) / temp[i-1]
        rets = np.diff(temp) / temp[:-1]

        s0 = temp[-1]

        # Sample mean and std (ddof=1 to match Julia Statistics.std)
        mu = np.mean(rets)
        sigma = np.std(rets, ddof=1)
        if sigma == 0 or np.isnan(sigma):
            # degenerate case: no volatility → vscore not meaningful
            vscores.append(0.0)
            continue

        drift = mu + 0.5 * sigma**2

        # noise: cumsum(randn(EPOCH, EXT), dims=2)
        noise = np.cumsum(np.random.randn(EPOCH, EXT), axis=1)

        # time steps 1..EXT as row vector, like (1:EXT)'
        timesteps = np.arange(1, EXT + 1, dtype=float)[None, :]  # shape (1, EXT)

        # paths = s0 * exp(σ * noise + drift * (1:EXT)')
        paths = s0 * np.exp(sigma * noise + drift * timesteps)

        # count of points where paths > s0
        sum_exceed = np.sum(paths > s0)

        vscores.append(sum_exceed / (EPOCH * EXT))

    v = np.array(vscores, dtype=float)
    v_mean = v.mean()
    v_std = v.std(ddof=1) if v.size > 1 else 0.0
    if v_std == 0 or np.isnan(v_std):
        return np.nan

    v_norm = (v - v_mean) / v_std

    # Return the "current" vscore (last one, like last element of Julia return)
    return float(v_norm[-1])


# ---------------------------------------------------------
# 2. Live VScore Engine
# ---------------------------------------------------------
class LiveVScoreEngine:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        initial_cash: float = 1000.0,
        buy_threshold: float = -2.0,
        sell_threshold: float = 2.0,
        cooldown_minutes: int = 30,
        lookback: int = 200,  # how many closes to keep
        obs: int = 100,
        epoch: int = 1000,
        ext: int = 20,
        paper: bool = True,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol

        # Strategy params
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.position = 0.0  # shares
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.cooldown = timedelta(minutes=cooldown_minutes)

        self.lookback = lookback
        self.obs = obs
        self.epoch = epoch
        self.ext = ext

        self.price_buffer = deque(maxlen=lookback)
        self.last_trade_time: datetime | None = None
        self.last_vscore: float | None = None

        # Alpaca clients
        self.trading_client = TradingClient(
            api_key,
            secret_key,
            paper=paper,
        )

        self.stream = StockDataStream(
            api_key=api_key,
            secret_key=secret_key,
            # Default feed is DataFeed.IEX for free accounts
        )

        # Subscribe to minute bars for symbol
        self.stream.subscribe_bars(self.on_bar, symbol)

    # -----------------------------------------------------
    # Order helpers
    # -----------------------------------------------------
    def _can_trade_now(self, t: datetime) -> bool:
        if self.last_trade_time is None:
            return True
        return (t - self.last_trade_time) >= self.cooldown

    def _log_state(self, t: datetime, price: float):
        equity = self.cash + self.position * price

        if self.last_vscore is None or np.isnan(self.last_vscore):
            v_str = "nan"
        else:
            v_str = f"{self.last_vscore:.3f}"

        print(
            f"[{t.isoformat()}] {self.symbol} close={price:.4f} "
            f"vscore={v_str} "
            f"pos={self.position:.4f} cash={self.cash:.2f} equity={equity:.2f}"
        )


    def _submit_market_order(self, side: OrderSide, qty: float):
        if qty <= 0:
            return

        order = MarketOrderRequest(
            symbol=self.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            resp = self.trading_client.submit_order(order_data=order)
            print(f"Submitted {side.name} order for {qty} {self.symbol}: id={resp.id}")
        except Exception as e:
            print(f"Order submission failed: {e}")

    def _handle_signal(self, t: datetime, price: float, vscore: float):
        # Long-only: buy when vscore < buy_threshold, sell when > sell_threshold
        if not self._can_trade_now(t):
            return

        # SELL signal
        if vscore > self.sell_threshold and self.position > 0:
            qty = self.position
            print(
                f"[{t.isoformat()}] SELL signal: vscore={vscore:.3f}, "
                f"qty={qty:.4f}, price={price:.4f}"
            )
            self._submit_market_order(OrderSide.SELL, qty)

            # Naive portfolio update at bar close price
            self.cash += qty * price
            self.position = 0.0
            self.last_trade_time = t

        # BUY signal
        elif vscore < self.buy_threshold and self.position == 0 and self.cash > 0:
            qty = self.cash / price
            print(
                f"[{t.isoformat()}] BUY signal: vscore={vscore:.3f}, "
                f"qty={qty:.4f}, price={price:.4f}"
            )
            self._submit_market_order(OrderSide.BUY, qty)

            # Naive portfolio update
            self.position = qty
            self.cash = 0.0
            self.last_trade_time = t

    # -----------------------------------------------------
    # Stream callback
    # -----------------------------------------------------
    async def on_bar(self, bar):
        """
        Async callback for StockDataStream minute bars.

        'bar' is an alpaca.data.models.bars.Bar instance:
        - bar.symbol
        - bar.timestamp (datetime)
        - bar.close (float)
        """
        t: datetime = bar.timestamp
        # Make timezone-aware consistent (Alpaca already uses UTC)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)

        price = float(bar.close)
        self.price_buffer.append(price)

        prices = np.array(self.price_buffer, dtype=float)
        v = compute_vscore(prices, OBS=self.obs, EPOCH=self.epoch, EXT=self.ext)
        self.last_vscore = v

        v_str = f"{v:.3f}" if not np.isnan(v) else "nan"
        print(
            f"[BAR] {t.isoformat()} {bar.symbol} close={price:.4f}, vscore={v_str}"
        )


        # Only act when vscore is valid
        if not np.isnan(v):
            self._handle_signal(t, price, v)

        # Optional: log current portfolio state
        self._log_state(t, price)

    # -----------------------------------------------------
    # Run
    # -----------------------------------------------------
    def run(self):
        """
        Start the websocket event loop and block.
        """
        print(f"Starting LiveVScoreEngine for {self.symbol}")
        print("Press Ctrl+C to stop.")
        try:
            self.stream.run()
        except KeyboardInterrupt:
            print("Stopping stream...")
            # stream.run() handles its own loop; on Ctrl+C we just exit


# ---------------------------------------------------------
# 3. Entry point
# ---------------------------------------------------------
def main():
    # Expect keys in env vars for safety
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        )

    symbol = "SPY"  # change to ASST / HIVE / etc. if you like

    engine = LiveVScoreEngine(
        api_key=api_key,
        secret_key=secret_key,
        symbol=symbol,
        initial_cash=1000.0,
        buy_threshold=-2.0,
        sell_threshold=2.0,
        cooldown_minutes=30,
        lookback=300,
        obs=100,
        epoch=1000,
        ext=20,
        paper=True,
    )

    engine.run()


if __name__ == "__main__":
    main()
