# Data Directory

This directory is used for caching datasets and storing preprocessed data for LoRA experiments.

## Datasets Used

### Primary Dataset: Alpaca Instruction Following
- **Source**: `tatsu-lab/alpaca` from Hugging Face Datasets
- **Type**: Instruction-following dataset derived from Stanford Alpaca
- **Size**: ~52K instruction-response pairs
- **Format**: Each example contains `instruction`, `input`, and `output` fields
- **Usage**: Fine-tuning language models to follow instructions

### Data Splits

For our experiments, we use:
- **Training**: First 1,000 examples (for initial baseline testing)
- **Evaluation**: Examples 1,000-1,200 (200 examples for evaluation)
- **Full Dataset**: Available for scaling up experiments

## Directory Structure

```
data/
├── README.md              # This file
├── raw/                   # Raw downloaded datasets (gitignored)
├── processed/             # Preprocessed/tokenized data (gitignored) 
└── cache/                 # HuggingFace datasets cache (gitignored)
```

## Data Processing Pipeline

1. **Loading**: Download from HuggingFace Hub using `datasets` library
2. **Formatting**: Apply instruction template to create consistent format
3. **Tokenization**: Convert text to tokens using model-specific tokenizer
4. **Caching**: Save processed data for faster subsequent loads

## Data Format

### Raw Alpaca Format
```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
```

### Processed Format (after template application)
```
### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris.
```

## Privacy and Ethics

- **Dataset Source**: Stanford Alpaca dataset is publicly available
- **License**: Check original dataset license before commercial use
- **Content**: May contain biases present in the source data
- **Usage**: Intended for research purposes in parameter-efficient fine-tuning

## Disk Space Requirements

- **Raw Dataset Cache**: ~100MB for full Alpaca dataset
- **Tokenized Data**: ~200MB (depends on tokenizer and sequence length)
- **Model Cache**: 1-15GB (depends on base model size)

## Configuration

Dataset settings are controlled in `experiments/configs/base_config.yaml`:

```yaml
data:
  dataset_name: "tatsu-lab/alpaca"
  max_length: 512
  train_split: "train[:1000]"      # Adjust for different experiment sizes
  eval_split: "train[1000:1200]"
  instruction_template: "### Instruction:\n{instruction}\n\n### Response:\n{output}"
```

## Scaling Experiments

To run larger experiments:

1. **Increase data size**: Modify `train_split` to `"train[:10000]"` or larger
2. **Use full dataset**: Set `train_split` to `"train[:-2000]"` and `eval_split` to `"train[-2000:]"`
3. **Add validation split**: Create separate validation set for hyperparameter tuning

## Notes

- All data files are automatically gitignored to prevent committing large datasets
- Downloaded data is cached locally to avoid repeated downloads
- Preprocessing is done on-the-fly but can be cached for repeated experiments 