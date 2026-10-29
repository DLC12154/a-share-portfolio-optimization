# 数据说明

本目录复用第一项研究冻结的Baostock历史数据：

- `csi300_prices_historical.parquet`：481只历史成分股，2019-07-01至2025-12-31后复权日线；
- `csi300_historical_membership.csv`：291个五日历史成分快照，共87,300条成员记录；
- `historical_dataset_manifest.json`：数据源、时间范围与SHA-256清单。

价格文件较大，默认被 `.gitignore` 排除。首次运行应把上述文件放入 `data/raw/`。数据本身已经复制到本项目，当前无需重新下载。

数据限制：Baostock历史成分和行情属于供应商重建数据，本项目无法独立证明所有历史修订。
