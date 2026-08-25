from mistralai import Mistral

from eval import evaluate
from generate_examples import generate_examples
from retacred_utils import Model_Family, Open_Source_API_Mode


def run_eval_retacred(model_family, client, model, tokenizer, experiment_name, open_source_api_mode, synthesize_examples):
    if synthesize_examples:
        generate_examples(model_family, client, model, tokenizer, experiment_name, open_source_api_mode)

    dev_results = []
    n_values = [0, 1, 5, 10]
    for n in n_values:
        tacred_file = "dev.0.01.json"
        dev_result = evaluate(model_family, client, model, tokenizer, tacred_file, n, experiment_name, open_source_api_mode)
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
    final_f1 = evaluate(model_family, client, model, tokenizer, tacred_file, n, experiment_name, open_source_api_mode)
    print(f"Final F1: {final_f1}")


def main():
    synthesize_examples = False # no need because it is already done

    api_key = "" # put your mistral.ai API key here

    client = Mistral(api_key=api_key)

    model = "mistral-large-2411"

    tokenizer = None

    model_family = Model_Family.OPEN_SOURCE

    experiment_name = model

    open_source_api_mode = Open_Source_API_Mode.MISTRAL

    run_eval_retacred(model_family, client, model, tokenizer, experiment_name, open_source_api_mode, synthesize_examples)


if __name__ == "__main__":
    main()

