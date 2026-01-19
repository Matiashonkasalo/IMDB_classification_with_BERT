# IMDb Sentiment Analysis with Knowledge Distillation

A PyTorch implementation of knowledge distillation for sentiment analysis, demonstrating how a lightweight CNN student model can learn from a powerful BERT teacher while achieving competitive performance at a fraction of the computational cost.

## Overview

This project showcases knowledge distillation applied to text classification. A compact TextCNN student model learns from soft predictions of a fine-tuned BERT teacher on the IMDb movie review dataset. The result is a fast, lightweight sentiment classifier that retains much of BERT's accuracy while being significantly faster and cheaper to deploy.

## Key Features

- **Task**: Binary sentiment classification (positive/negative reviews)
- **Dataset**: IMDb movie reviews (50k samples)
- **Teacher model**: Fine-tuned BERT-base
- **Student model**: TextCNN (Kim, 2014)
- **Training approach**: Soft label knowledge distillation
- **Frameworks**: PyTorch, Hugging Face Transformers
- **Experiment tracking**: MLflow with local logging

## What is Knowledge Distillation?

Knowledge distillation is a model compression technique that transfers knowledge from a large, accurate model (teacher) to a smaller, efficient model (student). Unlike traditional training with hard labels (0 or 1), the student learns from the teacher's soft probability distributions, which encode richer information about decision boundaries, uncertainty, and inter-class relationships.

**Benefits:**
- Smaller model size (10-100x reduction)
- Faster inference (5-50x speedup)
- Retention of 85-95% of teacher performance
- Easier deployment on resource-constrained devices

## Architecture
```
┌─────────────────────┐
│   IMDb Reviews      │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐  ┌─────────┐
│  BERT   │  │ TextCNN │
│(Teacher)│  │(Student)│
└────┬────┘  └────┬────┘
     │            │
     │  Soft      │  Hard
     │  Labels    │  Labels
     │            │
     └─────┬──────┘
           │
           ▼
    ┌──────────────┐
    │ Distillation │
    │     Loss     │
    └──────────────┘
```

## Project Structure
```
knowledge-distillation-imdb/
├── models/                      # Saved student checkpoints (gitignored)
│   ├── student_epoch_1.pt
│   ├── student_epoch_2.pt
│   └── student_epoch_3.pt
├── src/
│   ├── bert.py                  # BERT teacher wrapper
│   ├── data_loading.py          # IMDb dataset & DataLoader
│   ├── evaluate.py              # Validation metrics
│   ├── models.py                # TextCNN student architecture
│   ├── text_processing.py       # Tokenization & vocabulary
│   └── train.py                 # Training & distillation loop
├── requirements.txt
└── README.md

Note: mlruns/ and mlflow.db are generated locally and gitignored
```

## Model Components

### Teacher Model (`src/bert.py`)

A pretrained BERT-base model fine-tuned on IMDb sentiment classification:
```python
teacher = BertTeacher(device)
teacher_probs = teacher.predict_proba(texts)  # Soft predictions
```

**Characteristics:**
- 110M parameters
- Frozen during student training
- Outputs probability distributions over sentiment classes

### Student Model (`src/models.py`)

A lightweight TextCNN architecture inspired by Kim (2014):
```python
student = TextCNN(vocab_size=len(vocab))
probs = student(input_ids)
```

**Architecture:**
- Word embeddings (300-dim)
- Parallel 1D convolutions (kernel sizes: 3, 4, 5)
- Max-over-time pooling
- Sigmoid output layer

**Characteristics:**
- ~2M parameters (50x smaller than BERT)
- 10-20x faster inference
- CPU-friendly deployment

## Training Objective

The student is trained using a weighted combination of two loss terms:
```
L = α · L_labels + (1 - α) · L_teacher
```

Where:
- **L_labels**: Binary cross-entropy with ground truth labels
- **L_teacher**: Binary cross-entropy with teacher's soft predictions
- **α**: Hyperparameter balancing the two objectives (default: 0.5)
```python
loss = alpha * label_loss + (1 - alpha) * distill_loss
```

The soft predictions from the teacher provide richer supervisory signals than hard labels alone, helping the student learn more nuanced decision boundaries.

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/matiashonkasalo/IMDB_classification.git
cd IMDB_classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

For CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Training

Run the training script from the project root:
```bash
python src/train.py
```

**What happens during training:**
- IMDb subset (3,000 samples) loads for fast experimentation
- Teacher generates soft predictions on-the-fly
- Student learns from both teacher predictions and true labels
- Checkpoints save to `models/` after each epoch
- Metrics log locally via MLflow

### Monitoring Experiments

Launch the MLflow UI to visualize training metrics:
```bash
mlflow ui
```

Then navigate to: `http://localhost:5000`

You can view:
- Training and validation loss curves
- Accuracy over epochs
- Hyperparameter comparisons
- Model checkpoints

## Example Output
```
Epoch 1/3 | Train Loss: 0.4892 | Val Loss: 0.4217 | Val Acc: 82.44%
Epoch 2/3 | Train Loss: 0.4028 | Val Loss: 0.3891 | Val Acc: 84.56%
Epoch 3/3 | Train Loss: 0.3714 | Val Loss: 0.3652 | Val Acc: 85.89%

✓ Student model saved to models/student_epoch_3.pt
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Subset training** | Fast iteration during development; scales to full dataset |
| **Frozen teacher** | Ensures stable supervision throughout training |
| **TextCNN student** | Proven architecture, efficient, interpretable |
| **Binary cross-entropy** | Natural fit for binary classification |
| **MLflow tracking** | Reproducible experiments without committing artifacts |
| **Word-level tokenization** | Simpler vocabulary, faster than subword methods |


## Future Enhancements

- [ ] Temperature-scaled distillation for better calibration
- [ ] Larger student architectures (BiLSTM, Transformer-lite)
- [ ] Full IMDb dataset training (50k samples)
- [ ] Baseline comparison (student trained without distillation)
- [ ] Model compression (quantization, pruning)
- [ ] Production-ready inference API
- [ ] Deployment examples (ONNX, TorchScript)

## References

1. **Kim, Y. (2014).** [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
2. **Hinton, G., Vinyals, O., & Dean, J. (2015).** [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
3. **Maas, A. L., et al. (2011).** [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/) - IMDb Dataset



## Acknowledgments

Built with PyTorch, Hugging Face Transformers, and MLflow.
