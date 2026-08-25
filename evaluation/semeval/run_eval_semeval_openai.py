from openai import OpenAI

from eval_reduce_label_space import eval_reduce_label_space
from eval_final import eval_final
from generate_examples import generate_examples
from semeval_utils import Model_Family


def evaluate(model_family, client, model, tokenizer, dataset, n, experiment_name):
    eval_reduce_label_space(model_family, client, model, tokenizer, dataset, n, experiment_name)
    return eval_final(model_family, client, model, tokenizer, dataset, n, experiment_name)


def run_eval_semeval(model_family, client, model, tokenizer, experiment_name, synthesize_examples):
    if synthesize_examples:
        generate_examples(model_family, client, model, tokenizer, experiment_name)

    dev_results = []
    n_values = [0, 1, 5, 10]
    for n in n_values:
        dataset = "train.0.01"
        dev_result = evaluate(model_family, client, model, tokenizer, dataset, n, experiment_name)
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
    final_f1 = evaluate(model_family, client, model, tokenizer, dataset, n, experiment_name)
    print(f"\nFinal F1: {final_f1}")


def main():
    synthesize_examples = False # no need because it is already done

    api_key = "" # put your OpenAI API key here

    client = OpenAI(api_key=api_key)

    model_family = Model_Family.OPENAI

    #model = "gpt-4o-mini-2024-07-18"
    model = "gpt-4.1-2025-04-14"

    tokenizer = None

    experiment_name = model

    run_eval_semeval(model_family, client, model, tokenizer, experiment_name, synthesize_examples)


if __name__ == "__main__":
    main()

