from openai import OpenAI

from eval import evaluate
from tacrev_utils import Model_Family


def run_eval_tacrev(model_family, client, model, tokenizer, experiment_name):
    dev_results = []
    n_values = [0, 1, 5, 10]
    for n in n_values:
        tacred_file = "dev.0.01.json"
        dev_result = evaluate(model_family, client, model, tokenizer, tacred_file, n, experiment_name)
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
    final_f1 = evaluate(model_family, client, model, tokenizer, tacred_file, n, experiment_name)
    print(f"Final F1: {final_f1}")


def main():
    api_key = "" # put your OpenAI API key here

    client = OpenAI(api_key=api_key)

    model_family = Model_Family.OPENAI

    #model = "gpt-4o-mini-2024-07-18"
    model = "gpt-4.1-2025-04-14"

    tokenizer = None

    experiment_name = model

    run_eval_tacrev(model_family, client, model, tokenizer, experiment_name)


if __name__ == "__main__":
    main()

