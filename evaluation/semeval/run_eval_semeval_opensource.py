from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

from eval_reduce_label_space import eval_reduce_label_space
from eval_final import eval_final
from generate_examples import generate_examples
from semeval_utils import Model_Family


def evaluate(model_family, model, tokenizer, dataset, n, experiment_name):
    eval_reduce_label_space(model_family, None, model, tokenizer, dataset, n, experiment_name)
    return eval_final(model_family, None, model, tokenizer, dataset, n, experiment_name)


def run_eval_semeval(model_family, model, tokenizer, experiment_name, synthesize_examples):
    if synthesize_examples:
        generate_examples(model_family, None, model, tokenizer, experiment_name)

    dev_results = []
    n_values = [0, 1, 5, 10]
    for n in n_values:
        dataset = "train.0.01"
        dev_result = evaluate(model_family, model, tokenizer, dataset, n, experiment_name)
        dev_results.append(dev_result)

    max_index = dev_results.index(max(dev_results))
    n = n_values[max_index]

    print()
    print(n_values)
    print(dev_results)
    print("max_index:", max_index)
    print("n:", n)
    print()

    dataset = "test_subset"
    final_f1 = evaluate(model_family, model, tokenizer, dataset, n, experiment_name)
    print(f"\nFinal F1: {final_f1}")


def main():
    synthesize_examples = False # no need because it is already done

    token = "" # put your Hugging Face token here
    login(token=token)

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_family = Model_Family.LLAMA_31_8B

    # or choose your local model, for example
    #model_id = "google/gemma-2-9b-it"
    #model_id = "mistralai/Ministral-8B-Instruct-2410"
    #model_family = Model_Family.OPEN_SOURCE

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right' # to prevent warnings
    tokenizer.pad_token = tokenizer.eos_token # to avoid issues with padding

    experiment_name = model_id.split("/")[-1]

    run_eval_semeval(model_family, model, tokenizer, experiment_name, synthesize_examples)


if __name__ == "__main__":
    main()

