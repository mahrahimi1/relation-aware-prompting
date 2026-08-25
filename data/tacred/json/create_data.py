import json

def create_json_data(input_file_name, output_file_name, tacred_file_name):
    with open(input_file_name, "r") as f:
        ids = json.load(f)

    with open(tacred_file_name, "r") as f:
        examples = json.load(f)

    examples = [example for example in examples if example["id"] in ids]

    with open(output_file_name, "w") as f:
        json.dump(examples, f)

    print(f"Number of examples in {output_file_name}: {len(examples)}")

create_json_data("dev.0.01_ids.json", "dev.0.01.json", "dev.json")
create_json_data("test_subset_ids.json", "test_subset.json", "test.json")

