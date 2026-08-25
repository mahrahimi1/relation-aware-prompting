from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

from eval import evaluate
from tacrev_utils import Model_Family


def run_eval_tacrev(model_family, model, tokenizer, experiment_name):
    dev_results = []
    n_values = [0, 1, 5, 10]
    for n in n_values:
        tacred_file = "dev.0.01.json"
        dev_result = evaluate(model_family, None, model, tokenizer, tacred_file, n, experiment_name)
        dev_results.append(dev_result)

    max_index = dev_results.index(max(dev_results))
    n = n_values[max_index]

    print()
    print(n_values)
    print(dev_results)
    print("max_index:", max_index)
    print("n:", n)
    print()

    tacred_file = "test_subset.json"
    final_f1 = evaluate(model_family, None, model, tokenizer, tacred_file, n, experiment_name)
    print(f"Final F1: {final_f1}")


def main():
    token = "" # put your Hugging Face token here
    login(token=token)

    model_id = "mistralai/Ministral-8B-Instruct-2410" # choose your local open source model

    model_family = Model_Family.OPEN_SOURCE

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    experiment_name = model_id.split("/")[-1]

    run_eval_tacrev(model_family, model, tokenizer, experiment_name)


if __name__ == "__main__":
    main()

