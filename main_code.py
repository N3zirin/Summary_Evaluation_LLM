import re
from tqdm import tqdm, trange
from openai import OpenAI
from together import Together
import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from helpers import Summary_Ranking_Task_TLDR, Summary_Scorring_Task, Factual_Consistency_SumEdits, Summary_Ranking_Task_FIB, Factual_Consistency_FCSTS, Factual_Consistency_CoGenSum, Factual_Consistency_FactCC

parser = argparse.ArgumentParser(description="arguments")
parser.add_argument("--dataset_name", type=str, default="cogensumm", help="Dataset names for evaluation: cogensumm, factcc, polytope, summeval, xsumfaith, frank.")
parser.add_argument("--llm_provider", type=str, default="dp", help="Specify the model to be utilized for evaluation (qwen, gpt, dp, lg, llama)")
parser.add_argument("--trad_method", type=str, default="", help="Specify the model to be utilized for evaluation (qwen, gpt, dp, lg, llama).")
parser.add_argument("--model_name", type=str, default="deepseek-chat", help="Designation of the model for assessment")
parser.add_argument("--task", type=str, default="consistency", help="Task for assessment (e.g., consistency, ranking)")
parser.add_argument("--split", type=str, default='val', help="Division of the dataset for evaluative purposes (e.g., training, validation, testing)")
parser.add_argument('--type', type=str, default='COT', help='Evaluation type to conduct (COT, no_COT)')
args = parser.parse_args()


if __name__=="__main__":
    if args.task == "ranking" and args.dataset_name == "tldr":
        dataset = load_dataset('json', data_files = "DatasetsFolder/batch18.json", split="train")
        Summary_Ranking_Task_TLDR(dataset, args.model_name, args.llm_provider)
    elif args.task == "scoring":
        dataset = load_dataset('json', data_files="DatasetsFolder\model_annotations.aligned.paired.jsonl", split="train")
        Summary_Scorring_Task(dataset, model_name=args.model_name, llm_provider=args.llm_provider)
    elif args.task == "consistency" and args.dataset_name == "sumedits":
        dataset =  load_dataset('json', data_files="DatasetsFolder\summedits_podcast.json", split="train")
        Factual_Consistency_SumEdits(dataset, args.model_name, args.llm_provider)
    elif args.task == "ranking" and args.dataset_name == "fib":
        dataset = load_dataset("r-three/fib",split="test")
        Summary_Ranking_Task_FIB(dataset, args.model_name, args.llm_provider)
    elif args.task == "consistency" and args.dataset_name == "fcsts":
        dataset =  load_dataset("achandlr/FactualConsistencyScoresTextSummarization", split="train")
        Factual_Consistency_FCSTS(dataset, args.model_name, args.llm_provider)
    elif args.task == "consistency" and args.dataset_name == "cogensum":
        dataset =  load_dataset('json', data_files="DatasetsFolder\cogensumm_val.jsonl", split="train")
        Factual_Consistency_CoGenSum(dataset, args.model_name, args.llm_provider)
    elif args.task == "consistency" and args.dataset_name == "factcc":
        dataset =  load_dataset("mtc/factcc_annotated_eval_data", split="test")
        Factual_Consistency_FactCC(dataset, args.model_name, args.llm_provider)
    