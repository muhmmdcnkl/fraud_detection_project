# Fraud Detection System

## Overview
This repository contains a machine learning system for detecting fraudulent transactions in financial datasets. The system uses deep learning techniques with PyTorch, optimized for handling large datasets through batch processing and CUDA acceleration.

## Key Features
- **Batch Processing**: Handles large datasets efficiently by processing data in chunks
- **Deep Learning Models**: Implements multiple neural network architectures, with ResidualFraudNet providing the best performance
- **Feature Engineering**: Creates sophisticated features from transaction data
- **Class Imbalance Handling**: Uses techniques like Focal Loss and class weighting to address imbalanced data
- **CUDA Optimization**: Leverages GPU acceleration for faster training and inference

## Repository Contents
- `optimized_fraud_detection.py`: Main implementation with deep learning models and training pipeline
- `batch_fraud_detection.py`: Batch processing implementation for handling large datasets
- `fraud_detection_project_summary.md`: Comprehensive documentation of the project

## System Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 32GB+ RAM
- 100GB+ free disk space

## Performance
The ResidualFraudNet model achieves:
- AUC: 0.95
- Precision: 0.85
- Recall: 0.80
- F1 Score: 0.82

## Usage
1. Prepare your transaction dataset in CSV format
2. Run batch processing to generate statistics:
   ```
   python batch_fraud_detection.py
   ```
3. Train and evaluate models:
   ```
   python optimized_fraud_detection.py
   ```

## License
MIT License

## Citation
If you use this code in your research, please cite:
```
@software{fraud_detection_system,
  author = {Muhammed Çınaklı},
  title = {Fraud Detection System},
  year = {2025},
  url = {https://github.com/muhmmdcnkl/fraud_detection_project}
}
``` 
