# Human-MME

Official repository for "Human-MME: A Holistic Evaluation Benchmark for Human-Centric Multimodal Large Language Models"

## Overview

Human-MME is a comprehensive evaluation benchmark designed to assess the capabilities of Multimodal Large Language Models (MLLMs) in human-centric scenarios. It encompasses a wide range of tasks.


## Running the Benchmark

To run the benchmark, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/Yuan-Hou/Human-MME.git
cd Human-MME
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the datasets:

Download the datasets from [Human-MME_data.zip](https://huggingface.co/datasets/Yuanhou/Human-MME/blob/main/Human-MME_data.zip) and extract them into the root directory to maintain the following structure:
```
Human-MME/
├── final_qa/
├── final_labeling/
├── mllm_models/
├── benchmark.py
```

4. Implement your MLLM:

Implement your MLLM in `mllm_models/` directory by extending the `BaseModel` class. You should implement the `predict` method to handle the input and return the output. You can refer to the existing implementations for guidance.

Then, register your model in the `MODEL_NAME_MAP` dictionary in `benchmark.py`.

5. Run the benchmark:
```bash
python benchmark.py --model_name YourModelName
```

The default concurrency is set to 8. You can adjust it using the `--concurrency` flag.

If you get interrupted during the evaluation, you can resume it by adding the `--continuing` flag:
```bash
python benchmark.py --model_name YourModelName --continuing
```

6. Get the results:

After the evaluation is complete, the answers are saved in the `results/` directory with a json file named after your model in `results/result_YourModelName.json`. You can get the evaluation metrics by running:
```bash
python benchmark.py --calc_metrics results/result_YourModelName.json
```



