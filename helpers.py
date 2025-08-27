from collections import defaultdict, deque
import re
import os
import time
from flask.cli import load_dotenv
from openai import OpenAI
from together import Together
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm, trange
from sklearn.metrics import balanced_accuracy_score, classification_report
from template import Summary_Scorring_Prompt, Summary_Scorring_Prompt_CoT


load_dotenv()

def extract_answer_qwen(text):
  pattern1 = r'\bAnswer:\sconsistent\b'
  pattern2 = r'\bFinal\sAnswer:\sconsistent\b'
  pattern3 = r'\bAnswer:\s\*\*consistent\*\*'
  pattern4 = r'\*\*Final\sAnswer:\sconsistent\*\*'
  pattern5 = r'\*\*Final\sanswer:\sconsistent\*\*'
  pattern6 = r'\bAnswer:\s\*\*Consistent\*\*'
  pattern7 = r'\b\*\*Answer:\*\*\n\*\*Consistent\*\*'
  pattern8 = r'\*\*Final\sanswer:\sConsistent\*\*'
  pattern9 = r'\bAnswer:\nconsistent'
  pattern10 = r'\bAnswer:\nConsistent'
  pattern11 = r'\*\*Answer:\*\*\s\*\*Consistent\*\*'
  pattern12 = r'\*\*Answer:\*\*\sConsistent'
  pattern12 = r'\*\*Answer:\*\*\sconsistent'
  pattern13 = r'\*\*Answer:\sconsistent\*\*'
  pattern14 = r'\*\*Answer:\sConsistent\*\*'
  pattern15 = r'Answer:\s\s\n\*\*Consistent\*\*'
  pattern16 = r'(\*\*(c|C)onsistency\*\*){1}$'
  pattern17 = r'(\b(c|C)onsistent){1}$'
  pattern18 = r'(\*\*(c|C)onsistent)\*\*{1}$'
  pattern19 = r'\bAnswer:\sconsistent\b'
  pattern20 = r'\bFinal\sAnswer:\sconsistent\b'
  pattern21 = r'\bAnswer:\s\*\*consistent\*\*'
  pattern22 = r'\*\*Final\sAnswer:\sconsistent\*\*'
  pattern23 = r'\*\*Final\sanswer:\sconsistent\*\*'
  pattern24 = r'\bAnswer:\s\*\*Consistent\*\*'
  pattern25 = r'\b\*\*Answer:\*\*\n\*\*Consistent\*\*'
  pattern26 = r'\*\*Final\sanswer:\sConsistent\*\*'
  pattern27 = r'\bAnswer:\nconsistent'
  pattern28 = r'\bAnswer:\nConsistent'
  pattern29 = r'\*\*Answer:\*\*\s\*\*Consistent\*\*'
  pattern30 = r'\*\*Answer:\*\*\sConsistent'
  pattern31 = r'\*\*Answer:\*\*\sconsistent'
  pattern32 = r'\*\*Answer:\sconsistent\*\*'
  pattern33 = r'\*\*Answer:\sConsistent\*\*'
  pattern34 = r'Answer:\s\s\n\*\*Consistent\*\*'
  pattern35 = r'(\*\*(c|C)onsistency\*\*){1}$'
  pattern36 = re.compile(
    r'^\*\*Answer\*\*:\s*Consistent\.\s*\Z',  # \Z = absolute end of string
    re.MULTILINE | re.IGNORECASE
)

  if re.search(pattern1, text) or re.search(pattern2, text) or re.search(pattern3, text)or re.search(pattern4, text)\
  or re.search(pattern5, text) or re.search(pattern6, text) or re.search(pattern7, text) or re.search(pattern9, text)\
  or re.search(pattern10, text) or re.search(pattern11, text) or re.search(pattern12, text) or re.search(pattern13, text)\
  or re.search(pattern14, text) or re.search(pattern15, text) or re.search(pattern16, text) or re.search(pattern17, text)\
  or re.search(pattern18, text) or re.search(pattern19, text) or re.search(pattern20, text) or re.search(pattern21, text)or re.search(pattern22, text)\
  or re.search(pattern23, text) or re.search(pattern24, text) or re.search(pattern25, text) or re.search(pattern26, text)\
  or re.search(pattern27, text) or re.search(pattern28, text) or re.search(pattern29, text) or re.search(pattern30, text)\
  or re.search(pattern31, text) or re.search(pattern32, text) or re.search(pattern33, text) or re.search(pattern34, text)\
  or re.search(pattern35, text) or re.search(pattern36, text):
    return 1
  else:
    return 0
  


def initialize_clients(name='gpt'):
    gpt_client = OpenAI(api_key=os.getenv("gpt_api"))
    dp_client = OpenAI(api_key=os.getenv("deepseek_api"), base_url="https://api.deepseek.com")
    lg_client = Together(api_key=os.getenv("lg_api"))
    llama_client = Together(api_key=os.getenv("llama_api"))
    qwen_client = OpenAI(
        api_key=os.getenv("qwen_api"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    if name == 'gpt':
        return gpt_client
    elif name == 'dp':
        return dp_client
    elif name == 'lg':
        return lg_client
    elif name == 'llama':
        return llama_client
    else:
        return qwen_client

class RateLimiter:
  def __init__(self, max_requests=60, time_window=60):
    self.max_requests = max_requests
    self.time_window = time_window
    self.requests = deque()
  def wait_if_needed(self):
    now = time.time()
    while self.requests and self.requests[0] <= now - self.time_window:
      self.requests.popleft()
    if len(self.requests) >= self.max_requests:
        sleep_time = self.requests[0] + self.time_window - now + 0.1
        if sleep_time > 0:
          time.sleep(sleep_time)
    self.requests.append(now)
    
def Summary_Ranking_Task_TLDR(dataset, model_name = "gpt-4.1-mini", llm_provider = "gpt"):

    rate_limiter = RateLimiter(max_requests=59, time_window=60)
    client = initialize_clients(llm_provider)
    failed_requests = []
    pattern = r'<Answer>\*{0,2}A\*{0,2}</Answer>'
    prompt = """Decide which one of the following summary is consistent with the corresponding article.
        Note that consistency means all information in the summary is supported by the article.
        Explain your reasoning step-by-step and then give the answer in <Answer>(A or B)</Answer> tags:

        <Article>
        {document}
        </Article>

        <Summary A>
        {sum_a}
        </Summary A>
        <Summary B>
        {sum_b}
        </Summary B>
        """

    predictions, true_labels = [], []

    with trange(len(dataset)) as t:
        for i in t:
            max_retries = 3
            retry_count = 0
            sucess = False
            sums = [sum['text'] for sum in dataset[i]['summaries']]
            document = dataset[i]['info']['post']
            while retry_count < max_retries:
                try:
                    rate_limiter.wait_if_needed()
                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt.format(
                                document=document,
                                sum_a=sums[0],
                                sum_b=sums[1]
                            )}
                        ],
                        stream=False
                    )
                    response = response.choices[0].message.content
                    if re.search(pattern, response, re.MULTILINE):
                        predictions.append(0)
                        print("Summary A is better")
                    else:
                        predictions.append(1)
                    true_labels.append(dataset[i]['choice'])
                    print(f"Response: {response}, true_label: {dataset[i]['choice']}")
                    success = True
                    break 
                except Exception as e:
                    retry_count += 1
                    error_msg = f"Request failed for item {i}, attempt {retry_count}/{max_retries}: {str(e)}"
                    print(error_msg)
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    
            if not success:
                print(f"Failed to process item {i} after {max_retries} attempts. Skipping...")
                failed_requests.append(i)    

            if i % 5 == 0 and i > 0:
                t.set_postfix(acc=balanced_accuracy_score(predictions, true_labels))

def Factual_Consistency_Task(dataset, model_name = "gpt-4.1-mini", llm_provider = "gpt"):
    rate_limiter = RateLimiter(max_requests=59, time_window=60)
    predictions = []
    true_labels = []
    failed_requests = []

    prompt = """Decide if the following summary is consistent with the corresponding article.
      Note that consistency means all information in the summary is supported by the article.
      Do not give any reasoning, just answer (consistent or inconsistent) at the end:
      <Article>
      {document}
      </Article>

      <Summary>
      {summary}
      </Summary>
      Answer:"""

    client = initialize_clients(llm_provider)

    with trange(len(dataset)) as t:
        for i in t:
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries:
                try:
                    rate_limiter.wait_if_needed()
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {
                                "role": "user",
                                "content": prompt.format(
                                    document=dataset[i]['doc'],
                                    summary=dataset[i]['summary']
                                ),
                            },
                        ],
                        stream=False
                    )
                    prediction = extract_answer_qwen(response.choices[0].message.content)
                    predictions.append(prediction)
                    true_labels.append(dataset[i]['label'])

                    print(response.choices[0].message.content)
                    print('-' * 100)
                    print(f"Prediction: {prediction} True Label: {dataset[i]['label']}")

                    success = True
                    break

                except Exception as e:
                    retry_count += 1
                    error_msg = f"Request failed for item {i}, attempt {retry_count}/{max_retries}: {str(e)}"
                    print(error_msg)

                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)

            if not success:
                print(f"Failed to process item {i} after {max_retries} attempts. Skipping...")
                failed_requests.append(i)

            if i % 5 == 0 and i > 0 and len(predictions) > 0:
                t.set_postfix(
                    accuracy=balanced_accuracy_score(true_labels, predictions),
                    processed=len(predictions),
                    failed=len(failed_requests),
                    total=i+1
                )

    print(classification_report(true_labels, predictions, target_names=["Inconsistent", "Consistent"]))

def calculate_correlations_with_pandas(averages, df_ratings, metrics):
    """Calculate Pearson and Spearman correlations excluding missing values using pandas"""
    results = {}

    for metric in metrics:
        # Create a DataFrame with both expert averages and model ratings
        comparison_df = pd.DataFrame({
            'expert_avg': averages[metric],
            'model_rating': df_ratings[metric]
        })

        # Drop rows where either value is missing (None/NaN)
        clean_data = comparison_df.dropna()

        if len(clean_data) > 1:  # Need at least 2 data points for correlation
            pearson_corr, pearson_p = pearsonr(clean_data['expert_avg'], clean_data['model_rating'])
            spearman_corr, spearman_p = spearmanr(clean_data['expert_avg'], clean_data['model_rating'])

            results[metric] = {
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'n_samples': len(clean_data),
                'excluded_samples': len(comparison_df) - len(clean_data)
            }
        else:
            results[metric] = {
                'pearson_correlation': None,
                'pearson_p_value': None,
                'spearman_correlation': None,
                'spearman_p_value': None,
                'n_samples': len(clean_data),
                'excluded_samples': len(comparison_df) - len(clean_data)
            }

    return results


def Summary_Scorring_Task(dataset, output_file='correlation_results.csv', model_name='deepseek-chat', llm_provider='dp', type="COT"):
    """
      Evaluate correlation between LLM predictions and human annotations
    """
    print(f"Evaluating correlation with model: {model_name}")
    client = initialize_clients(llm_provider)
    EXPECTED_METRICS = ['coherence', 'consistency', 'fluency', 'relevance']
    ratings = defaultdict(list)
    pattern = re.compile(r"\*\*(\w+):\s*(\d+)\*\*")
    if type == "COT":
       prompt = Summary_Scorring_Prompt_CoT
    else:
       prompt = Summary_Scorring_Prompt
    print(f"Starting Correlation Evaluation with {model_name} and llm_provider {llm_provider} and prompt type {type}")
    try:
        with trange(len(dataset)) as t:
          for i in t:
            messages = [
                {"role": "system", "content": "You are a human annotator that rates the quality of summaries. You must provide a rating for Coherence, Consistency, Fluency, and Relevance on a scale of 1 to 5, in the format **Metric: Score**."},
                {"role": "user", "content": prompt.format(article=dataset[i]['text'], summary=dataset[i]['decoded'])}
            ]

            # A dictionary to hold the results for just this one item
            found_scores_for_item = {}

            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    stream=False,
                )
                response_text = response.choices[0].message.content
                #print(response_text)
                # Parse the response and populate the temporary dictionary
                pattern = re.compile(r"(?:\*\*)?(\w+):\s*(\d+)(?:\*\*)?", re.IGNORECASE)
                for match in pattern.finditer(response_text):
                    metric_name = match.group(1).lower()
                    score = int(match.group(2))
                    if metric_name in EXPECTED_METRICS:
                        found_scores_for_item[metric_name] = score

            except Exception as e:
                # Handle cases where the API call itself fails
                print(f"API call failed for item {i}: {e}")
                # The found_scores_for_item dictionary will remain empty

            # --- This is the key change ---
            # Ensure all lists grow by one for every item, using None for missing data.
            ratings['id'].append(dataset[i]['id'])
            for metric in EXPECTED_METRICS:
                # .get(key, default_value) is perfect for this.
                # It will get the score if found, otherwise it will return None.
                score = found_scores_for_item.get(metric, None)
                ratings[metric].append(score)

    finally:
        # --- Corrected DataFrame Creation ---
        # Convert the entire dictionary to a DataFrame at the end.
        # Pandas handles the dictionary of lists format perfectly.
        print("\nLoop finished or interrupted. Saving results...")
        df = pd.DataFrame(ratings)
        df.to_csv('Ratings.csv', index=False)
        print("Results saved to Ratings.csv")

    metrics = ['coherence', 'consistency', 'fluency', 'relevance']
    averages = {metric: [] for metric in metrics}

    for item in dataset:
        annotations = item.get('expert_annotations', [])
        if not annotations:
            for metric in metrics:
                averages[metric].append(None)
            continue
        num_annotations = len(annotations)
        for metric in metrics:
            total_score = sum(anno[metric] for anno in annotations)
            averages[metric].append(total_score / num_annotations)
    results = calculate_correlations_with_pandas(averages, df, metrics)
    print(results)
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return results_df