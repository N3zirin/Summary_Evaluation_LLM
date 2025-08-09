import re
from tqdm import tqdm, trange
from openai import OpenAI
from together import Together
from datasets import load_dataset
from sklearn.metrics import accuracy_score
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
  
dataset = load_dataset("json", data_files = "SummaCoz/polytope_val.jsonl", split = "train")
api_key="42e6a032337914a799289c823c282e2c270ef8c5d07dbaee5ff1dd71fef3f03d" # setting the global variable for api key for multiple usage
Llama_client = Together(api_key = api_key)

predictions = []
true_labels = []
with trange(len(dataset)) as t:
  for i in t:
    # response1 = lg_client.chat.completions.create(
    #   model="lgai/exaone-deep-32b",
    #   messages=[
    #       {"role": "system", "content": "You are a helpful assistant"},
    #       {"role": "user", "content": f"""Decide if the following summary is consistent with the corresponding article.
    #   Note that consistency means all information in the summary is supported by the article.
    #   Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
    #   <Article>
    #   {dataset['test'][i]['text']}
    #   </Article>

    #   <Summary>
    #   {dataset['test'][i]['claim']}
    #   </Summary>

    #   Answer:
    #   """},
    #   ],
    #   stream=False
    # )
    # response2 = llama_client.chat.completions.create(
    #   model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    #   messages=[
    #       {"role": "system", "content": "You are a helpful assistant"},
    #       {"role": "user", "content": f"""Decide if the following summary is consistent with the corresponding article.
    #   Note that consistency means all information in the summary is supported by the article.
    #   Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
    #   <Article>
    #   {dataset[i]['context']}
    #   </Article>

    #   <Summary>
    #   {dataset[i]['summary']}
    #   </Summary>

    #   Answer:
    #   """},
    #   ],
    #   stream=False
    # )
    # response3 = gpt_client.chat.completions.create(
    #   model="gpt-4.1-mini",
    #   messages=[
    #       {"role": "system", "content": "You are a helpful assistant"},
    #       {"role": "user", "content": f"""Decide if the following summary is consistent with the corresponding article.
    #   Note that consistency means all information in the summary is supported by the article.
    #   Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
    #   <Article>
    #   {dataset['test'][i]['text']}
    #   </Article>

    #   <Summary>
    #   {dataset['test'][i]['claim']}
    #   </Summary>

    #   Answer:
    #   """},
    #   ],
    #   stream=False
    # )
    # response4 = qwen_client.chat.completions.create(
    #   model="qwen-plus",
    #   messages=[
    #       {"role": "system", "content": "You are a helpful assistant"},
    #       {"role": "user", "content": f"""Decide if the following summary is consistent with the corresponding article.
    #   Note that consistency means all information in the summary is supported by the article.
    #   Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
    #   <Article>
    #   {dataset[i]['document']}
    #   </Article>

    #   <Summary>
    #   {dataset[i]['claim']}
    #   </Summary>

    #   Answer:
    #   """},
    #   ],
    #   stream=False
    # )
    response5 = Llama_client.chat.completions.create(
      model="lgai/exaone-deep-32b",
      messages=[
          {"role": "system", "content": "You are a helpful assistant"},
          {"role": "user", "content": f"""
    Evaluate if the following summary is consistent with the article by checking for:

        1. *FACTUAL ACCURACY*: Are all statements in the summary factually correct according to the article?
        2. *OMISSION CHECK*: Does the summary miss any crucial information, key details, or important context from the article?
        3. *COMPLETENESS*: Does the summary adequately represent the article's main points and essential details?

        Analyze step-by-step:
        - First, verify each claim in the summary against the source
        - Then, identify any important information from the article that's missing from the summary
        - Finally, assess if omissions significantly impact understanding

        Answer: consistent or inconsistent
        <document> {dataset[i]["document"]} </document>
        <summary>{dataset[i]["claim"]}</summary>
        Answer:
      """},
      ],
      stream=False
    )
    # lst = []
    # lst.append(extract_answer_qwen(response1.choices[0].message.content))
    # lst.append(extract_answer_qwen(response2.choices[0].message.content))
    # lst.append(extract_answer_qwen(response3.choices[0].message.content))
    # lst.append(extract_answer_qwen(response4.choices[0].message.content))

    # print(response.choices[0].message.content)
    # prediction = most_frequent(lst)

    prediction = extract_answer_qwen(response5.choices[0].message.content)
    predictions.append(prediction)
    true_labels.append(dataset[i]['label'])
    print(response5.choices[0].message.content)
    print('-'*100)
    print(f"""Prediction: {prediction} True Label: {dataset[i]['label']}""")
    if i%5 == 0 and i > 0:
      t.set_postfix(accuracy=accuracy_score(predictions, true_labels))