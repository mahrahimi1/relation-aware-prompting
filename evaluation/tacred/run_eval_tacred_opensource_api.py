from huggingface_hub import login, InferenceClient

from eval import evaluate
from generate_examples import generate_examples
from tacred_utils import Model_Family, Open_Source_API_Mode


def run_eval_tacred(model_family, client, model, tokenizer, experiment_name, open_source_api_mode, synthesize_examples):
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

    token = "" # put your Hugging Face token here

    # "auto" will automatically select the first provider available for the model, sorted by the user's order.
    # you can also choose any available provider for the model (nebius, novita, hf-inference, etc)
    provider = "auto"

    client = InferenceClient(
        provider=provider,
        api_key=token,
        #headers={"X-use-cache": "false"}
    )

    #model = "meta-llama/Llama-3.1-70B-Instruct"
    model = "google/gemma-3-27b-it"

    model_family = Model_Family.OPEN_SOURCE

    tokenizer = None

    experiment_name = model.split("/")[-1]

    open_source_api_mode = Open_Source_API_Mode.HUGGING_FACE

    run_eval_tacred(model_family, client, model, tokenizer, experiment_name, open_source_api_mode,synthesize_examples)


if __name__ == "__main__":
    main()

