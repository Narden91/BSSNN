# Benchmark Datasets for LTSF

This document lists the benchmark datasets used for Long-Term Time Series Forecasting (LTSF) experiments and their sources.

## Datasets

| Dataset | Description | Source | Note |
| :--- | :--- | :--- | :--- |
| **ETT** (ETTh1, ETTh2, ETTm1, ETTm2) | Electricity Transformer Temperature | [GitHub - zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) | Already present in `data/ETT` (partial) |
| **Electricity** (ECL) | Hourly electricity consumption of 321 clients | [Google Drive (Autoformer/LTSF-Linear)](https://drive.google.com/drive/folders/1Zjhx6HAbD9L9HwE-aK9Kq2V0bLgN7g7?) | Via `cure-lab/LTSF-Linear` |
| **Traffic** | Hourly traffic occupancy from 862 sensors | [Google Drive (Autoformer/LTSF-Linear)](https://drive.google.com/drive/folders/1Zjhx6HAbD9L9HwE-aK9Kq2V0bLgN7g7?) | Via `cure-lab/LTSF-Linear` |
| **Weather** | 21 meteorological indicators every 10 min | [Google Drive (Autoformer/LTSF-Linear)](https://drive.google.com/drive/folders/1Zjhx6HAbD9L9HwE-aK9Kq2V0bLgN7g7?) | Via `cure-lab/LTSF-Linear` |
| **Exchange** | Daily exchange rates of 8 countries | [Google Drive (Autoformer/LTSF-Linear)](https://drive.google.com/drive/folders/1Zjhx6HAbD9L9HwE-aK9Kq2V0bLgN7g7?) | Via `cure-lab/LTSF-Linear` |
| **ILI** | Weekly influenza-like illness data | [Google Drive (Autoformer/LTSF-Linear)](https://drive.google.com/drive/folders/1Zjhx6HAbD9L9HwE-aK9Kq2V0bLgN7g7?) | Via `cure-lab/LTSF-Linear` |

## Download Instructions

1.  **ETT**: Clone the [ETDataset](https://github.com/zhouhaoyi/ETDataset) repository or download `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv` and place them in `data/ETT/`.
2.  **Others**: Go to the [Autoformer/LTSF-Linear Google Drive](https://drive.google.com/drive/folders/1Zjhx6HAbD9L9HwE-aK9Kq2V0bLgN7g7?).
    *   Download `electricity.csv` -> `data/electricity/`
    *   Download `traffic.csv` -> `data/traffic/`
    *   Download `weather.csv` -> `data/weather/`
    *   Download `exchange_rate.csv` -> `data/exchange_rate/`
    *   Download `national_illness.csv` -> `data/illness/`

**Note**: Ensure the file names match what the data loader expects (usually standardized to `name.csv`).
