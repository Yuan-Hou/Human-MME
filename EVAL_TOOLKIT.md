# Human-MME Evaluation Toolkit

This document provides a comprehensive guide for using `benchmark.py` to evaluate Multimodal Large Language Models (MLLMs) on the Human-MME benchmark.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Arguments](#command-line-arguments)
- [Supported Models](#supported-models)
- [Question Types & Answer Formats](#question-types--answer-formats)
- [Running Evaluations](#running-evaluations)
- [Batch Export Mode](#batch-export-mode)
- [Metrics Calculation](#metrics-calculation)
- [Output Files](#output-files)
- [Adding Custom Models](#adding-custom-models)

---

## Overview

The `benchmark.py` script is the main evaluation toolkit for the Human-MME benchmark. It supports:

- **Concurrent evaluation** of MLLMs with configurable parallelism
- **Multiple question types** including bounding box, multiple choice, fill-in-the-blank, sequence ordering, and more
- **Automatic answer parsing** with regex-based extraction
- **Comprehensive metrics** including IoU, accuracy, BERTScore, cosine similarity, and ranking metrics
- **Resume capability** to continue interrupted evaluations
- **Batch export** for API-based batch processing

---

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Required Packages

Key dependencies include:
- `bert_score` - For semantic text similarity
- `sentence_transformers` - For embedding-based evaluation
- `Pillow` - For image processing
- `rich` - For progress display
- `regex` - For answer parsing
- `openai` / `zhipuai` - For API-based models

---

## Quick Start

### Basic Evaluation

Run a complete benchmark evaluation:

```bash
python benchmark.py --qa_dir ./final_qa --model_name qwen2.5-vl-72b --concurrency 8
```

### Calculate Metrics Only

If you already have results, calculate metrics:

```bash
python benchmark.py --qa_dir ./final_qa --calc_metrics ./results/results_qwen2.5-vl-72b.json
```

---

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--qa_dir` | string | `./final_qa` | Directory containing QA pairs (each subdirectory should have a `qa.json` file) |
| `--model_name` | string | *required* | Name of the model to benchmark (must be in `MODEL_NAME_MAP`) |
| `--concurrency` | int | `8` | Number of concurrent threads for parallel processing |
| `--model_params` | string | `''` | Additional model parameters in JSON format |
| `--export_batch` | int | `0` | Batch size for batch prediction export (generates JSONL files) |
| `--continuing` | flag | `False` | Continue from existing results file if it exists |
| `--calc_metrics` | string | `None` | Path to existing results JSON file for metrics calculation only |

---

## Question Types & Answer Formats

The benchmark supports various question and answer types, each with specific format instructions:

### Answer Types

#### 1. Bounding Box (`bbox`)
Localize an object in the image with coordinates.

**Format:**
```
Analyze: <your analysis>
Answer: [x1,y1,x2,y2]
```

**Metrics:** IoU (Intersection over Union), Accuracy (IoU ≥ 0.5)

#### 2. True/False Bounding Box (`tf_bbox`)
Localize an object if it exists, otherwise indicate absence.

**Format:**
```
Analyze: <your analysis>
Answer: [x1,y1,x2,y2], if no match, answer [-1,-1,-1,-1]
```

**Metrics:** IoU, Accuracy, TF correctness

#### 3. Multiple Choice (`choice`)
Select one option from A/B/C/D.

**Format:**
```
Analyze: <your analysis>
Answer: A/B/C/D
```

**Metrics:** Accuracy

#### 4. Fill-in-the-Blank (`blank`)
Provide a short, concise answer.

**Format:**
```
Analyze: <your analysis>
Answer: <your final answer in a short and concise expression>
```

**Metrics:** BERTScore F1, Cosine Similarity, Keyword Coverage, Composite Score

#### 5. True/False Blank (`tf_blank`)
Provide an answer if applicable, otherwise indicate unknown.

**Format:**
```
Analyze: <your analysis>
Answer: <your final answer in a short and concise expression>, if no match, answer unknown
```

**Metrics:** BERTScore, Cosine Similarity, TF correctness

#### 6. Open HOI (Human-Object Interaction) (`open_hoi`)
Identify an object name and its bounding box.

**Format:**
```
Analyze: <your analysis>
Name: <name of the object>
Box: [x1,y1,x2,y2]
```

**Metrics:** IoU, Object Accuracy, Composite Score

#### 7. Sequence Ordering (`sequence`)
Order four options in the correct sequence.

**Format:**
```
Analyze: <your analysis>
First: A/B/C/D
Second: A/B/C/D
Third: A/B/C/D
Fourth: A/B/C/D
```

**Metrics:** Accuracy, Kendall's Tau, Spearman's Rho, nDCG@4

#### 8. Double Choice (`double_choice`)
Predict both past and future events.

**Format:**
```
Analyze: <your analysis>
Past: A/B/C/D
Future: A/B/C/D
```

**Metrics:** Past Accuracy, Future Accuracy, Overall Accuracy

---

## Running Evaluations

### Standard Evaluation

```bash
python benchmark.py \
    --qa_dir ./final_qa \
    --model_name qwen2.5-vl-72b \
    --concurrency 8
```

### With Custom Model Parameters

Pass additional parameters to the model constructor:

```bash
python benchmark.py \
    --qa_dir ./final_qa \
    --model_name qwen2.5-vl-72b \
    --model_params '{"api_base": "http://localhost:8000/v1", "api_key": "your-key"}'
```

### Resume Interrupted Evaluation

If an evaluation was interrupted, use `--continuing` to resume:

```bash
python benchmark.py \
    --qa_dir ./final_qa \
    --model_name qwen2.5-vl-72b \
    --continuing
```

This will load existing results from `results/results_<model_name>.json` and only process unanswered questions.

---

## Batch Export Mode

For models that support batch API processing, export questions to JSONL format:

```bash
python benchmark.py \
    --qa_dir ./final_qa \
    --model_name gpt-4-vision \
    --export_batch 1000
```

This generates:
- `batch/<model_name>/images/` - Processed images
- `batch/<model_name>/batch_<model_name>_<batch_no>.jsonl` - Batch request files

The JSONL format is compatible with OpenAI's Batch API and similar services.

---

## Metrics Calculation

### Calculate Metrics from Existing Results

```bash
python benchmark.py \
    --qa_dir ./final_qa \
    --calc_metrics ./results/results_qwen2.5-vl-72b.json
```

### Available Metrics

| Metric | Description | Applicable Types |
|--------|-------------|------------------|
| `accuracy` | Exact match or threshold-based accuracy | All |
| `iou` | Intersection over Union for bounding boxes | bbox, tf_bbox, open_hoi |
| `bert_f1` | BERTScore F1 for text similarity | blank, tf_blank, open_hoi |
| `cos_sim` | Cosine similarity of sentence embeddings | blank, tf_blank, open_hoi |
| `kw_coverage` | Keyword coverage ratio | blank, tf_blank, open_hoi |
| `composite_score` | Weighted combination (0.5×BERT + 0.3×cos + 0.2×kw) | blank, tf_blank, open_hoi |
| `tau` | Kendall's Tau (normalized to [0,1]) | sequence |
| `rho` | Spearman's Rho (normalized to [0,1]) | sequence |
| `ndcg` | nDCG@4 for ranking quality | sequence |
| `analysis_len` | Length of analysis text | All |

---

## Output Files

The toolkit generates several output files:

### During Evaluation

| File | Description |
|------|-------------|
| `results/results_<model_name>.json` | Raw results with responses and ground truth |

### After Metrics Calculation

| File | Description |
|------|-------------|
| `results/results_<model_name>_metrics.txt` | Summary metrics by category |
| `results/results_<model_name>_detailed_metrics.json` | Detailed per-question metrics |

### Results JSON Structure

```json
{
    "path/to/question": {
        "response": {
            "raw": "Full model response...",
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "analysis": "Model's analysis..."
        },
        "ground_truth": [0.1, 0.2, 0.3, 0.4],
        "q_type": "localization",
        "a_type": "bbox"
    }
}
```

---

## Adding Custom Models

To add a new model, create a class that extends `BaseModel`:

```python
# mllm_models/my_model.py
from mllm_models.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name)
        # Initialize your model here
        
    def predict(self, image, question):
        """
        Args:
            image: PIL.Image object (RGB)
            question: String containing the question with format instructions
            
        Returns:
            String containing the model's response
        """
        # Implement your prediction logic
        return response
    
    def concurrency(self):
        """
        Returns:
            True if the model supports concurrent requests, False otherwise
        """
        return True
```

Then register it in `benchmark.py`:

```python
from mllm_models.my_model import MyModel

MODEL_NAME_MAP = {
    # ... existing models ...
    "my-model": MyModel,
}
```
