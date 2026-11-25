# SwitchIt Traces

SwitchIt records what an agent did and matches those actions to their observed outcomes. The collectors save canonical events, source hashes, and audit results under `Sova-Kit-data/traces/` alongside the kit. [Saved traces](../../../Sova-Kit-data/traces/)

| Collector | Evidence collected |
| --- | --- |
| [EnterpriseOps-Gym](collect_enterpriseops.py) | Published Teams transcript, upstream task, and checks reproduced from tool receipts |
| [TradesBench](collect_tradesbench.py) | Call traces and their final outcomes from a published simulator database |

SwitchUp uses reviewed outcomes and source event IDs to build correction and replay cases. [Task state](../../SwitchUp/TASK_STATE.md)
