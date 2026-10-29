# 验证记录

验证日期：2026-10-30
Python：3.12.15  
解释器：`C:\Users\ad\.conda\envs\teach\python.exe`

## 输入完整性

- 历史成分快照：291个。
- 历史成分记录：87,300条。
- 历史成员并集：481只。
- 行情记录：731,507条，无重复date-code键。
- 行情与成分文件SHA-256与冻结清单一致。

## 单元测试

执行：

```powershell
C:\Users\ad\.conda\envs\teach\python.exe -m unittest discover -s tests -v
```

结果：11项测试全部通过，覆盖：

- 权重归一化与单票上限；
- 逆波动率权重；
- 异构Beta、不可行Beta与换手约束；
- 求解器报告成功后仍需通过二次约束检查；
- Beta分子和分母使用同一配对样本；
- 风险惩罚是否真实改变配置；
- 换手受限目标是否逐步退出旧持仓；
- 协方差对称半正定；
- 缺失收益资产剔除；
- 次日开盘交易和成本扣除。

## 全流程

执行：

```powershell
C:\Users\ad\.conda\envs\teach\python.exe run.py
```

结果：

- 291个信号日全部生成目标组合；
- 目标权重记录181,928条，交易记录226,668条；
- `optimized`和`optimized_utility`各成功291次，回退0次；
- 每个执行日、每个策略目标权重和为1；
- 优化组合最大单票权重5%；
- 最大事前年化波动率22%；
- 最大目标单边换手50%；
- 事前Beta位于相对当期合格成分股等权代理的0.80至1.20；
- 净值全部为正，交易成本和成交名义金额非负。

输出审计：

```powershell
C:\Users\ad\.conda\envs\teach\python.exe scripts\validate_outputs.py
```

结果：`PASS`。

## 结果文件

- `data/processed/performance.csv`：分阶段组合绩效；
- `data/processed/target_weights.csv.gz`：所有策略目标权重；
- `data/processed/optimizer_diagnostics.csv`：每期风险、换手、合格成分池等权Beta和求解状态；
- `data/processed/trades.csv.gz`：完整交易日志；
- `data/processed/rebalances.csv`：调仓、换手、成本和阻塞统计；
- `data/processed/daily_nav.csv`：每日净值；
- `figures/`：时间外净值、回撤、换手、分年收益和事前风险图。

## 结论边界

修正Beta定义后，优化组合在2024–2025时间外区间没有战胜全市场等权净基准，也没有跑赢同持股规模的Top 20/25等权组合。2025年复合信号Mean RankIC为-0.0060，信号方向已经失效。因此结果不支持“复合Alpha跨年度稳定有效”，也不支持“优化器具有稳定独立增量”。
