# SalesForecasting Deep Learning for Supply Chain Optimization in Retail

## Project Overview
This project explores the use of Deep Learning (DL) models to improve supply chain efficiency through sales forecasting. 
The aim is to provide insights into sales data so that businesses can make informed decisions, reduce waste, and minimize supply chain costs.

## Problem Statement
Sales forecasting is a long-standing challenge in supply chain management. Traditional models like ARIMA and ETS often fail to capture non-linear patterns and long-term dependencies. 
Recent transformer-based models, particularly BART, show promise in overcoming these challenges.

## Dataset
- **Primary Dataset:** [DataCo Supply Chain Dataset (Mendeley)](https://data.mendeley.com/datasets/8gx2fvg2k6/3)
- **Features:**
  - Customer info, order details, product info, sales & profit, shipping, geography, and time features
- **Other Datasets Explored:**
  - M5 Forecasting
  - Retail dataset from Kaggle

## Models Explored
- **LSTM / GRU:** Poor generalization, negative or near-zero R².
- **Transformer Encoder:** Failed to capture variance in target data.
- **XGBoost:** Good performance with R² ~87%, but limited trend awareness.
- **BART (facebook/bart-base):** Best performance using a sequence-to-sequence forecasting approach.

## Final Method: BART-based Seq2Seq Forecasting
- Pretrained **facebook/bart-base** (139M parameters)
- 6-layer encoder + 6-layer decoder, 12 attention heads
- Input: 30-day sales history
- Output: 90-day forecast
- Fine-tuned using Hugging Face's `Seq2SeqTrainer`

### Training Details
- Epochs: 10  
- Learning Rate: 3e-5  
- Weight Decay: 0.01  
- Batch Size: 4  
- Beam Search: 4 beams  
- Max Generation Length: 256 tokens  

## Results & Evaluation
- **ROUGE-1:** 39.93  
- **ROUGE-2:** 26.13  
- **ROUGE-L:** 30.31  
- **BLEU:** 20.18  
- **BERTScore F1:** 90.73  

**Takeaways:**
- Outperformed LSTM, GRU, and XGBoost in capturing sales trends.
- Effectively modeled semantic patterns in imbalanced, noisy retail data.
- Slightly smoothed sharp fluctuations but showed stable convergence.


