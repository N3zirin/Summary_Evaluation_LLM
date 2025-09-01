from unsloth import FastLanguageModel
import pandas as pd
import numpy as np
import re 
from tqdm import tqdm, trange
import torch
from datasets import load_dataset
from vllm import SamplingParams
from sklearn.metrics import balanced_accuracy_score

max_seq_length = 4096 #limit for the input
lora_rank = 256

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8
)

model = FastLanguageModel.get_peft_model(
    model, 
    r =lora_rank,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'dpwn_proj'], #qwen llm model layers choice query, k metrics, value metrics
    lora_alpha = lora_rank * 2, # learning rate, alpha is doubled, it is trainig faster
    use_gradient_checkpointing = "unsloth",
    random_state = 9363,
    )

reasoning_start = "<start_working_out>" # Acts as <think>
reasoning_end   = "<end_working_out>"   # Acts as </think>
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
system_prompt

"""We create a simple chat template below. Notice `add_generation_prompt` includes prepending `<start_working_out>` to guide the model to start its reasoning process."""

chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"
# Replace with out specific template:
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

CONSISTENT_PATTERNS = re.compile(
    r'(?:'
    r'\bAnswer:\s*\*{0,2}[Cc]onsistent\*{0,2}\b|'
    r'\bFinal\s+[Aa]nswer:\s*\*{0,2}[Cc]onsistent\*{0,2}\b|'
    r'\*{0,2}Answer\*{0,2}:\s*\*{0,2}[Cc]onsistent\*{0,2}|'
    r'\*{0,2}Final\s+answer:\s*\*{0,2}[Cc]onsistent\*{0,2}|'
    r'\*{0,2}[Cc]onsistency\*{0,2}$|'
    r'\b[Cc]onsistent$|'
    r'Answer:\s*\n\*{0,2}[Cc]onsistent\*{0,2}'
    r')',
    re.IGNORECASE | re.MULTILINE
)

answer_patterns = [
        r'answer:\s*\*?\*?(consistent|inconsistent)\*?\*?',  # Handle **consistent** format
        r'conclusion:\s*\*?\*?(consistent|inconsistent)\*?\*?', 
        r'final answer:\s*\*?\*?(consistent|inconsistent)\*?\*?',
        r'therefore.*?(consistent|inconsistent)',
        r'(consistent|inconsistent)(?:\s*[.]?\s*$)',
    ]

def extract_answer_qwen(text: str) -> int:
    return 1 if CONSISTENT_PATTERNS.search(text) else 0
def add_labels(x):
    label = extract_answer_qwen(x["solutions"])
    return label
def extract_answer(response):
    response_lower = response.lower().strip()
    for pattern in answer_patterns:
        match = re.search(pattern, response_lower, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    if 'inconsistent' in response_lower:
        return 'inconsistent'
    elif 'consistent' in response_lower:
        return 'consistent'
    return None

def truncate_text(text, max_tokens=1500):
    words = text.split()
    max_words = int(max_tokens * 0.75)
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + "..."
    return text

def create_prompt_with_truncation(item, system_prompt, tokenizer, max_length=2048):
     base_prompt = f"""
    Evaluate if the following summary is consistent with the article. Note that consistency means all information in the summary is supported by the article. Explain your reasoning step-by-step first, and the answer (consistent or inconsistent) at the end:
        <document> {{article}} </document>
        <summary>{{summary}}</summary>
        Answer:
        """
     messages_template = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": base_prompt}
     ]

     base_text = tokenizer.apply_chat_template(
         messages_template, 
         add_generation_prompt = True,
         tokenize = False
     )
     base_tokens = len(tokenizer.encode(base_text))

     generation_butter = 512
     available_tokens = max_length - base_tokens - generation_butter

     article_tokens = int(available_tokens * 0.8) #80% of the tokens to article and the rest for the llm generated text
     summary_tokens = available_tokens - article_tokens

     truncated_article = truncate_text(item['context'], article_tokens)
     truncated_summary = truncate_text(item['summary'], summary_tokens)

     final_prompt = base_prompt.replace("{article}", truncated_article).replace("{summary}", truncated_summary)
     messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": final_prompt}
     ]
     return messages


def run_single_experiment(seed, num_samples = 1000, max_length = 2048):
    dataset = load_dataset("achandlr/FactualConsistencyScoresTextSummarization", split = "train")
    dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    predictions = []
    true_labels = []
    skipped_items = 0

    with trange(len(dataset), desc = f"Seed {seed}") as t:
        for i in t:
            item = dataset[i]

            try:
                messages = create_prompt_with_truncation(item, system_prompt, tokenizer, max_length)
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt = True,
                    tokenize = False,
                )
                token_count = len(tokenizer.encode(text))
                if token_count > max_length:
                    print(f"Warning: Item {i} still too long ({token_count} tokens), skipping...")
                    skipped_items += 1
                    continue
                sampling_params = SamplingParams(
                    temperature = 1.0,
                    top_k=50,
                    max_tokens = 512,
                )

                output = model.fast_generate(
                    text,
                    sampling_params = sampling_params,
                    lora_request = None #model.load_lora("outputs/consistency_grpo/final_model_last"),
                )[0].outputs[0].text

                answer = extract_answer(output)
                dataset[i]['model_answer'] = answer

                true_labels.append(item['label'])
                predictions.append(1 if answer == 'consistent' else 0)
                
                if i % 10 == 0 and i > 0:
                    current_acc = balanced_accuracy_score(true_labels, predictions)
                    t.set_postfix(acc = current_acc, skipped=skipped_items, tokens = token_count)
            
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                skipped_items += 1
                continue
    
    if len(predictions) > 0:
        final_accuracy = balanced_accuracy_score(true_labels, predictions)
        print(f"Seed {seed} - Processed: {len(predictions)}, Skipped: {skipped_items}, Accuracy: {final_accuracy}")
        return final_accuracy, len(predictions), skipped_items
    else:
        print(f"Seed {seed} - No item were successfully processed!")
        return None, 0, skipped_items
    
def run_multiple_experiments(seeds, num_samples=1000, max_length = 2048):
    accuracies = []
    processed_counts = []
    skipped_counts = []

    print(f"Running {len(seeds)} experiments with seeds: {seeds}")
    print(f"Using {num_samples} sample per experiment")
    print(f"Max context length: {max_length} tokens\n")

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Running experiment with seed: {seed}")
        print(f"{'='*50}")

        result = run_single_experiment(seed, num_samples, max_length)
        accuracy, processed, skipped = result

        if accuracy is not None:
            accuracies.append(accuracy)
            processed_counts.append(processed)
            skipped_counts.append(skipped)

            print(f"Experiment {seed} completed successfully")

            if len(accuracies) > 1:
                current_mean = np.mean(accuracies)
                current_std = np.std(accuracies, ddof=1)
                print(f"Running mean: {current_mean:.4f} +- {current_std:.4f}")
        else:
            print(f"Exeriment {seed} failed - no valid predictions")

    if len(accuracies) == 0:
        print("\nERROR: No experimens completed successfully!")
        return None
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies, ddof=1)
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)

    total_processed = sum(processed_counts)
    total_skipped = sum(skipped_counts)
    avg_processed = np.mean(processed_counts)
    avg_skipped = np.mean(skipped_counts)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Seeds used: {seeds}")
    print(f"Successful experiments: {len(accuracies)}/{len(seeds)}")
    print("Samples per experiment: {num_samples}")
    print(f"Max context length: {max_length} tokens")
    print(f"\nProcessing Statistics:")
    print(f"Total items processed: {total_processed}")
    print(f"Total items skipped: {total_skipped}")
    print(f"Average processed per experiment: {avg_processed:.1f}")
    print(f"Average skipped per experiment: {avg_skipped:.1f}")
    print(f"Success rate: {avg_processed/(avg_processed + avg_skipped)*100:.1f}%")

    print(f"\nAccuracy Result:")
    print(f"Individual accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"standard deviation: {std_accuracy:.4f}")
    print(f"Min accuracy: {min_accuracy:.4f}")
    print(f"Max accuracy: {max_accuracy:.4f}")
    print(f"Range: {max_accuracy - min_accuracy:.4f}")
    print(f"\nConfidence interval (+-1 std): {mean_accuracy:.4f} +- {std_accuracy:.4f}")
    print(f"95% confidence interval (+-1.96 std): {mean_accuracy:.4f}+- {1.96 * std_accuracy:.4f}")

    return {
        'accuracy': accuracies,
        'mean': mean_accuracy,
        'std': std_accuracy,
        'min': min_accuracy,
        'max': max_accuracy,
        'seeds': seeds[:len(accuracies)], # only successful seeds
        'processed_counts': processed_counts,
        'skipped_counts': skipped_counts,
        'total_processed': total_processed,
        'total_skipped': total_skipped
    }

#example usage

if __name__ == "__main__":
    seeds = [3407, 43, 4352, 999, 396, 2003, 15703, 5084]
    results = run_multiple_experiments(seeds, num_samples=1000, max_length=4096)
    if results is not None:
        import json
        with open('accuracy_results.json', 'w') as f:
            json.dump(results, f, indent = 2)
        print(f"\nResults saved to 'accurcy_results.json'")

        print(f"\n{'='*40}")
        print("SUMMARY")
        print(f"{'='*40}")
        print(f"Mean Accuracy: {results['mean']:.4f} +- {results['std']:.4f}")
        print(f"Processing Success Rate: {results['total_processed']/(results['total_processed'] + results['total_processed'])*100:.1f}%")