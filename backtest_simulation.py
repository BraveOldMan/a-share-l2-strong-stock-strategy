import polars as pl
import numpy as np

class StatefulPortfolioManager:
    """
    V14 状态机持久化断点传导 (Stateful Execution)
    抛弃了旧版基于未来 T+2 标签的短视静态回测，变成一个像实盘一样不断滚动的真正基金。
    """
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        stop_loss_pct: float = -0.05,
        trailing_stop_pct: float = 0.08,  # "只要趋势在，坚决不卖出", 回撤 8% 才触发止盈
        max_positions: int = 5,
        commission_rate: float = 0.00025,
        stamp_duty_rate: float = 0.0005,
        base_slippage_bps: float = 0.0015,
        max_slippage_bps: float = 0.0150,
    ):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}       # code -> { volume, cost_price, highest_mtm, entry_date }
        self.pending_buys = []    # list of dicts: {'code', 'intended_portion'}
        
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_positions = max_positions
        
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.base_slippage_bps = base_slippage_bps
        self.max_slippage_bps = max_slippage_bps
        self.current_date = None

    def process_daily_market(self, df_today: pl.DataFrame, current_date: str):
        """
        1. 盘前: 执行堆积的开仓排单 (T+1 开盘买入)
        2. 盘中: 监控现有持仓，触发追踪止盈/止损 (T+1 及以上方可卖出)
        """
        self.current_date = current_date
        if df_today.height == 0:
            return

        # 获取市场行情字典 O(1)
        daily_dict = {row['万得代码']: row for row in df_today.to_dicts()}

        # =========== 1. 尝试执行昨日的待买入信号 (Pending Buys) ===========
        if self.pending_buys:
            for order in self.pending_buys:
                code = order['code']
                if code not in daily_dict:
                    continue
                    
                row = daily_dict[code]
                day_open = row.get('day_open', 0.0)
                if day_open < 1e-3:
                    continue
                
                # [V14] 一字涨停板逼空过滤
                # 近似判断方式：如果全天高低点完全重合，或者开盘就被顶死
                day_high = row.get('day_high', 1.0)
                day_low = row.get('day_low', 0.0)
                if day_high == day_low and day_open > 0:
                    print(f"  [Portfolio BUY REJECTED] {code} Limit-Up locked all day.")
                    continue
                
                # [V15] "反包陷阱"过滤：拒绝追顶（高开超过 8%）和过度水下（低开低于 -2%）
                pre_close = row.get('pre_close', 0.0)
                if pre_close > 0.0:
                    gap_up = (day_open - pre_close) / pre_close
                    if gap_up > 0.08:
                        print(f"  [Portfolio BUY REJECTED] {code} Gap-Up {gap_up*100:.1f}% > 8%, avoiding trap.")
                        continue
                    if gap_up < -0.02:
                        print(f"  [Portfolio BUY REJECTED] {code} Gap-Down {gap_up*100:.1f}% < -2%, weak start.")
                        continue
                
                intended_cash = order['intended_portion']
                
                # 微观流动性深度滑点惩罚
                impact_ratio = 0.0
                dynamic_slippage = self.base_slippage_bps
                if 'max_limit_up_order_amt' in row and row['max_limit_up_order_amt'] > 0:
                    max_order = row['max_limit_up_order_amt']
                    # 放松对买入资金的硬性截断，改为使用惩罚性滑点平滑消化
                    impact_ratio = intended_cash / max_order
                    dynamic_slippage = self.base_slippage_bps + (impact_ratio ** 2) * self.max_slippage_bps
                    dynamic_slippage = min(dynamic_slippage, self.max_slippage_bps)

                if self.cash < intended_cash:
                    intended_cash = self.cash
                    
                if intended_cash < 1000:
                    continue

                actual_entry_price = day_open * (1.0 + dynamic_slippage)
                commission = intended_cash * self.commission_rate
                
                net_invested = intended_cash - commission
                volume = net_invested / actual_entry_price
                
                self.cash -= intended_cash
                
                self.positions[code] = {
                    'volume': volume,
                    'cost_price': actual_entry_price,
                    'highest_mtm': actual_entry_price, # 盘中会继续计算
                    'entry_date': current_date,
                }
                w_pct = intended_cash / self.initial_capital * 100
                print(f"  [Portfolio BUY] {code} | W={w_pct:.1f}% | Price: {actual_entry_price:.2f}")

            self.pending_buys.clear()

        # =========== 2. T+N 持仓维护 (动态止盈/止损) ===========
        for code in list(self.positions.keys()):
            if code not in daily_dict:
                continue
            
            row = daily_dict[code]
            pos = self.positions[code]
            
            day_open = row.get('day_open', 0.0)
            day_high = row.get('day_high', 0.0)
            day_low = row.get('day_low', 0.0)
            day_close = row.get('day_close', 0.0)
            
            if day_open < 1e-3:
                continue

            # 更新历史最高价 (Highest MTM - Mark-to-Market)
            # 无期极刑，"只设止损，利润奔跑"
            if day_high > pos['highest_mtm']:
                pos['highest_mtm'] = day_high
            
            # [A股真实T+1限制] 当日买入的标的，不享受当日盘中离场权力
            if pos['entry_date'] == current_date:
                continue
                
            # 只要趋势在，不设上限；8% 移动止盈，成本下 5% 的硬止损
            exit_trigger_price = max(
                pos['cost_price'] * (1 + self.stop_loss_pct),
                pos['highest_mtm'] * (1 - self.trailing_stop_pct)
            )
            
            # [V15] 固定止盈线：涨幅 > 15% 直接落袋为安
            take_profit_price = pos['cost_price'] * 1.15
            
            is_stopped_out = day_low <= exit_trigger_price
            is_take_profit = day_high >= take_profit_price
            
            # 是否有效击穿或撞线？
            if is_stopped_out or is_take_profit:
                
                # 确定卖出价格与原因
                if is_take_profit and not is_stopped_out:
                    # 冲破止盈且没有击穿下方，盘中按固定止盈线或开盘即顶出走货
                    sell_price = max(day_open, take_profit_price)
                    reason = "Static Take-Profit (+15%)"
                else:
                    # 击穿止损了！
                    sell_price = min(day_open, exit_trigger_price)
                    reason = f"Trailing Stop ({self.trailing_stop_pct*100:.0f}%)" if exit_trigger_price == pos['highest_mtm'] * (1 - self.trailing_stop_pct) else "Hard Stop-Loss"
                
                # 防范一字板跌停锁死无量跌停 (Limit-Down Trapped)
                if day_high == day_low and sell_price < pos['cost_price'] * 0.99:
                    print(f"  [Portfolio TRAPPED] {code} | 一字跌停板封死，止损失败。")
                    continue
                
                # 能够成交流动性，执行斩仓
                actual_exit_price = sell_price * (1 - self.base_slippage_bps)
                gross_return = pos['volume'] * actual_exit_price
                exit_cost = gross_return * (self.commission_rate + self.stamp_duty_rate)
                net_return = gross_return - exit_cost
                
                profit = net_return - (pos['volume'] * pos['cost_price'])
                pct_return = profit / (pos['volume'] * pos['cost_price'])
                
                self.cash += net_return
                del self.positions[code]
                
                color = "\033[31m" if profit < 0 else "\033[32m"
                print(f"  [Portfolio SELL] {code} | {reason} | PnL: {color}{pct_return*100:.2f}%\033[0m")

    def generate_target_orders(self, df_top3: pl.DataFrame, current_market_regime: int = -1, market_breadth_safe: bool = True):
        """
        盘后基于最新预测分发弹药：
        - 市场暴跌状态熔断：停止发车
        - 动态解耦：如果池子里有这只票，绝不重复买入。剩下的新席位给新票。
        """
        if current_market_regime == 0:
            print("\033[33m  [Portfolio] 大盘极端恶劣 (Regime 0) - 熔断防守，今日不分配新订单！\033[0m")
            return
            
        if not market_breadth_safe:
            print("\033[31m  [Portfolio] 全市场宽基动能崩盘 (Breadth 死叉) - 强制空仓，今日不分配新订单！\033[0m")
            return

        if df_top3.height == 0:
            return

        # V16 强行否决补丁（VPIN过滤）已回滚，保持原有 V15 空缺直接分派逻辑。

        if df_top3.height == 0:
            return

        # [V14] 去重清算防重复吃进锁 (Deduplication Check)
        current_holdings = set(self.positions.keys())
        df_new_signals = df_top3.filter(~pl.col("万得代码").is_in(list(current_holdings)))
        
        if df_new_signals.height == 0:
            return # 名单上全是老将，已经发过车了

        # 看看还有多少持仓空位
        vacancies = self.max_positions - len(self.positions)
        if vacancies <= 0:
            return # 满仓运作中
            
        # 裁剪到最大限额数量
        df_new_signals = df_new_signals.head(vacancies)
        
        # [V19] 自适应资金平铺：将所有的剩余闲水平摊给可用空位，但单只股票最高不超过总资产的 35%
        alloc_per_ticker = self.cash / max(vacancies, 1)
        max_alloc_cap = self.initial_capital * 0.35
        if alloc_per_ticker > max_alloc_cap:
            alloc_per_ticker = max_alloc_cap
        
        simulated_cash = self.cash
        for row in df_new_signals.to_dicts():
            code = row["万得代码"]
            
            # 最高买入单票配额，但绝不透支超出账面现金
            alloc = min(alloc_per_ticker, simulated_cash)
            if alloc < 1000:
                break
                
            simulated_cash -= alloc
            
            self.pending_buys.append({
                'code': code,
                'intended_portion': alloc,
            })

    def get_equity(self, df_today: pl.DataFrame) -> float:
        """
        盘点：持仓市值 + 闲水资金
        """
        daily_dict = {row['万得代码']: row for row in df_today.to_dicts()}
        market_value = 0.0
        
        for code, pos in self.positions.items():
            if code in daily_dict:
                last_price = daily_dict[code].get('day_close', pos['cost_price'])
                market_value += pos['volume'] * last_price
            else:
                market_value += pos['volume'] * pos['cost_price']
                
        return self.cash + market_value

    def get_summary(self):
        return f"[Fund] 现金: {self.cash:.2f} | 持仓支数: {len(self.positions)} | 待接盘单: {len(self.pending_buys)}"
