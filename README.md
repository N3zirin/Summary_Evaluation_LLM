# Factual Consistency of Text Summarization  

## MSc Dissertation: *“Factual Consistency in Text Summarization Tasks”*  

This repository is part of an MSc dissertation project that investigates the **factual consistency of text summarization**. The focus is on evaluating how well summaries reflect their source texts across multiple domains, including news, medical, podcasts, Reddit, and others.  

### Project Scope  
- Evaluation of factual accuracy and consistency in text summarization  
- Experiments conducted with multiple datasets from different domains  
- Use of both traditional evaluation methods and large language model (LLM)–based approaches  
- Tasks include summary ranking and consistency checking  

### Repository Contents  
- Source code for experiments  
- Datasets used in the study  
- Documentation of methods and results  

---  
This project aims to provide insights into the strengths and limitations of current approaches for evaluating factual consistency in text summarization.  


# Usage Guide

## Command Line Arguments

| Argument | Type | Default | Description | Available Options |
|----------|------|---------|-------------|-------------------|
| `--dataset_name` | str | `cogensumm` | Dataset name for evaluation | `cogensumm`, `factcc`, `polytope`, `summeval`, `xsumfaith`, `frank`, `tldr`, `fib`, `sumedits`, `fcsts` |
| `--llm_provider` | str | `dp` | LLM provider for evaluation | `qwen`, `gpt`, `dp`, `lg`, `llama` |
| `--trad_method` | str | `""` | Traditional method for evaluation | (specify traditional method name) |
| `--model_name` | str | `deepseek-chat` | Specific model designation | Model-specific names (e.g., `deepseek-chat`) |
| `--task` | str | `consistency` | Evaluation task type | `consistency`, `ranking`, `scoring` |
| `--split` | str | `val` | Dataset split for evaluation | `train`, `val`, `test` |
| `--type` | str | `COT` | Evaluation prompting type | `COT`, `no_COT` |

## Supported Task-Dataset Combinations

| Task | Dataset | Description | Data Source |
|------|---------|-------------|-------------|
| `consistency` | `cogensumm` | Factual consistency evaluation | Local file: `DatasetsFolder/cogensumm_val.jsonl` |
| `consistency` | `factcc` | Factual consistency evaluation | HuggingFace: `mtc/factcc_annotated_eval_data` |
| `consistency` | `sumedits` | Factual consistency evaluation | Local file: `DatasetsFolder/summedits_podcast.json` |
| `consistency` | `fcsts` | Factual consistency evaluation | HuggingFace: `achandlr/FactualConsistencyScoresTextSummarization` |
| `ranking` | `tldr` | Summary ranking task | Local file: `DatasetsFolder/batch18.json` |
| `ranking` | `fib` | Summary ranking task | HuggingFace: `r-three/fib` |
| `ranking` | `sumpairs` | Summary ranking task | Local file: `DatasetsFolder/benchmark_data.json` |
| `ranking and consistency`|`fib, fcsts, factcc, sumpairs, cogensum, sumedits, tldr`|SummacConv, NER, SummacZs, Google Flan-T5: Traditional methods for ranking and consistency evaluation|-|
| `scoring` |`sumeval`| Summary scoring task | Local file: `DatasetsFolder/model_annotations.aligned.paired.jsonl` |
| `scoring` |`summeval`| BART, BERT, ROUGE, METEOR, BLEU, SACREBLEU Traditional methods | Local file: `DatasetsFolder/model_annotations.aligned.paired.jsonl`|

## Usage Examples

```bash
# Factual consistency evaluation with CoGenSum dataset using DeepSeek
python main.py --dataset_name cogensumm --llm_provider dp --model_name deepseek-chat --task consistency --type COT

# Summary ranking with FIB dataset using GPT
python main.py --dataset_name fib --llm_provider gpt --model_name gpt-4 --task ranking --type no_COT

# Factual consistency evaluation with FactCC dataset using LLaMA
python main.py --dataset_name factcc --llm_provider llama --model_name llama-7b --task consistency --split test

# Summary scoring task
python main.py --task scoring --llm_provider qwen --model_name qwen-chat
```

## Notes

- Ensure all required dataset files are present in the `DatasetsFolder/` directory for local datasets
- HuggingFace datasets will be automatically downloaded when first accessed
- The `--type` parameter controls whether Chain-of-Thought (CoT) prompting is used
- Some combinations may require specific model configurations or API keys


```markdown
## GRPO Fine-Tuning

This repository includes advanced fine-tuning capabilities using Generalized Reinforcement Learning from Process Outcomes (GRPO) for improving factual consistency evaluation.

### GRPO Training Features

- **Multi-Reward System**: Uses 4 specialized reward functions:
 - `check_consistency_format`: Rewards proper reasoning structure
 - `check_consistency_answer`: Validates correctness against ground truth  
 - `check_reasoning_quality`: Evaluates depth and quality of reasoning
 - `check_evidence_usage`: Rewards proper source material referencing

- **Custom Chat Template**: Implements structured reasoning with special tokens:
 - `<start_working_out>`: Beginning of reasoning process
 - `<end_working_out>`: End of reasoning process
 - `<SOLUTION>`: Final answer section

### GRPO Training Setup

Run GRPO training by modifying paths in the script as needed:
`python grpo_train.py`

### Key Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | `unsloth/Qwen3-4B-Base` | Foundation model for fine-tuning |
| LoRA Rank | 128 | Low-rank adaptation parameter |
| Max Sequence Length | 4096 | Maximum context length |
| Learning Rate | 5e-6 | GRPO training learning rate |
| Batch Size | 1 | Per-device training batch size |
| Generations | 8 | Number of generations per prompt |
| Max Steps | 3000 | Total training steps |

### Training Data Format

The GRPO trainer expects data with the following structure:
- `prompt`: The factual consistency evaluation task
- `answer`: Ground truth label (consistent/inconsistent) 
- Proper reasoning traces in the expected format

### Model Outputs

After GRPO training, models produce structured responses:
```
<start_working_out>
[Detailed reasoning about consistency...]
<end_working_out>
<SOLUTION>
consistent/inconsistent
</SOLUTION>
```
```