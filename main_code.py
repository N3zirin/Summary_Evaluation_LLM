import re
from tqdm import tqdm, trange
from openai import OpenAI
from together import Together
import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from helpers import Summary_Ranking_Task_TLDR, Summary_Scorring_Task, Factual_Consistency_Task

parser = argpars.ArgumentParser(description="arguments")
parser.add_argument("--dataset_name", type=str, default="cogensumm", help="Name of the dataset to evaluate(cogensumm, factcc, polytope, summeval, xsumfaith, frank)")
parser.add_argument("--llm_provider", type=str, default="dp", help="Name of the model to use for evaluation(qwen, gpt, dp, lg, llama)")
parser.add_argument("--trad_method", type=str, default="", help="Name of the traditional method to use for evaluation (summac, bartscore, ner_consistency)")
parser.add_argument("--model_name", type=str, default="deepseek-chat", help="Name of the model to use for evaluation")
parser.add_argument("--task", type=str, default="consistency", help="Task to evaluate (e.g., consistency, ranking, )")
parser.add_argument("--split", type=str, default='val', help="Split of the dataset to use for evaluation (e.g., train, val, test)")
parser.add_argument('--type', type=str, default='COT', help='Type of evaluation to perform (COT, no_COT)')
args = parser.parse_args()


if __name__=="__main__":
    if args.task == "ranking" and args.dataset_name == "TLDR":
        dataset = load_dataset('json', data_files = "DatasetsFolder/batch18.json", split="train")
        Summary_Ranking_Task_TLDR(dataset, args.model_name, args.llm_provider)
    elif args.task == "scoring":
        dataset = load_dataset('json', data_files="DatasetsFolder\model_annotations.aligned.paired.jsonl", split="train")
        Summary_Scorring_Task(dataset, model_name=args.model_name, llm_provider=args.llm_provider)
    elif args.task == "consistency":
        dataset =  load_dataset('json', data_files="DatasetsFolder\summedits_podcast.json", split="train")
        Factual_Consistency_Task(dataset, args.model_name, args.llm_provider)