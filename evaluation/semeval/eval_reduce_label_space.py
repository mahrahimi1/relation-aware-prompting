import json
import time
import sys
import os
import datetime
import statistics
from ast import literal_eval

import openai
import backoff

from transformers import pipeline
import torch
from tqdm import tqdm
import numpy as np

from semeval_utils import get_relation_set, get_relation_set_extended, get_options, get_relation_set_string,\
                          get_definition, get_examples, Model_Family, Open_Source_API_Mode


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_completion_openai(prompt, client, model):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


def get_completion_open_source(prompt, pipe):
    messages = [{"role": "user", "content": prompt}]

    outputs = pipe(
        messages,
        max_new_tokens=256,
        do_sample=False,
        #temperature = 0,
        temperature=None, top_k=None, top_p=None,
        pad_token_id=pipe.tokenizer.eos_token_id
    )

    response = outputs[0]["generated_text"][-1]["content"]
    return response


def get_completion_open_source_api(prompt, client, model):
    messages = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0
    )

    return completion.choices[0].message.content


def get_completion_mistral_api(prompt, client, model):
    messages = [{"role": "user", "content": prompt}]

    chat_response = client.chat.complete(
        model= model,
        messages = messages,
        temperature=0,
        max_tokens=256,
    )

    return chat_response.choices[0].message.content


def get_completion(open_source_api_mode, model_family, prompt, pipe, model):
    if model_family == Model_Family.OPENAI:
        return get_completion_openai(prompt, pipe, model)

    elif (model_family == Model_Family.OPEN_SOURCE) or (model_family == Model_Family.LLAMA_31_8B):
        if open_source_api_mode is None:
            return get_completion_open_source(prompt, pipe)

        elif open_source_api_mode == Open_Source_API_Mode.HUGGING_FACE:
            return get_completion_open_source_api(prompt, pipe, model)

        elif open_source_api_mode == Open_Source_API_Mode.MISTRAL:
            return get_completion_mistral_api(prompt, pipe, model)

        else:
            raise Exception(f"Open source API mode '{open_source_api_mode}' not supported.")

    else:
        raise Exception(f"Model Family '{model_family}' Not Supported.")


def get_clean_entity_type(entity_type):
    clean_entity_type = entity_type.replace("_", " ")
    return clean_entity_type


def trim_response(response):
    if response[:8] == "```json\n":
        return response[8:-4]

    elif response[:10] == "```python\n":
        return response[10:-4]

    else:
        return response


def match_response_to_label(response, relation_set):
    option_labels, options, _ = get_options(relation_set)

    if response in option_labels:
        return option_labels[response]

    if (response[-1] == '.') and (response[:-1] in option_labels):
        return option_labels[response[:-1]]

    if response in options:
        return options[response]

    option_labels_keys = list(option_labels.keys())
    options_keys = list(options.keys())
    for i in range(len(option_labels)):
        option_label = option_labels_keys[i]
        clean_relation = options_keys[i]
        if response == f"{option_label}. {clean_relation}":
            return options[clean_relation]

    split = response.split('.')
    if len(split) == 2:
        if split[0] in option_labels:
            try:
                return options[split[1].strip()]
            except:
                return option_labels[split[0]]

    if len(split) > 2:
        if split[0] in option_labels:
            return option_labels[split[0]]

    raise Exception("Response cannot be matched to any label.")


def get_patterns_string(patterns):
    if len(patterns) == 1:
        return f'"{patterns[0]}"'

    if len(patterns) == 2:
        return f'"{patterns[0]}" and "{patterns[1]}"'

    s = ""
    for pattern in patterns[:-1]:
        s += f'"{pattern}", '
    s += "and " + f'"{patterns[-1]}"'

    return s


def get_prompt(example, n, experiment_name, model_family):
    e1 = example["e1"]
    e2 = example["e2"]
    context = example["context"]

    relation_set = get_relation_set()

    _, _, option_list_string = get_options(relation_set)
    relation_set_string = get_relation_set_string(relation_set)

    prompt = "In the task of relation extraction, you are given a sentence and a pair of entities in the sentence. " \
             "The goal is to select the relation between the two entities from a predefined set of candidate relations.\n\n" \
    
    prompt += f"The predefined relations are {relation_set_string}.\n\n"

    prompt += 'The definition of each relation is as follows. Note that in relation examples or relation instances, X and Y are replaced with actual words.\n'

    for relation in relation_set:
        if relation == "Other":
            continue
        relation_label = relation

        filename = "json/relations_restrictions_v1.json"
        relation_definition = get_definition(relation, filename)

        prompt += f'\n{relation_label}:\n{relation_definition}\n'

    prompt += '\n\nIf none of the above relations holds between the two entities, we output "Other".'

    for relation in relation_set:
        if relation == "Other":
            continue
        relation_label = relation

        if n > 0:
            if n == 1:
                prompt += f'\n\nBelow is an example sentence for the "{relation_label}" relation:'
            else:
                prompt += f'\n\nBelow are {n} example sentences for the "{relation_label}" relation:'

            file_name = f"json/relations_patterns_v4_{experiment_name}.json"
            patterns = get_examples(relation, file_name, n)
            for pattern in patterns:
                prompt += f"\n{pattern}"


    prompt += f'\n\nNow, based on the above definitions and restrictions, which of the following relations may hold ' \
              f'between "{e1}" and "{e2}" or between "{e2}" and "{e1}" in the following sentence? You must select at least 3 ' \
              f'of the relations that might hold. Be as permissive as possible when selecting the relations that might hold.\n\n'

    prompt += f"Sentence: {context}\n\n"

    prompt += f"Options:\n{option_list_string}\n"

    prompt += f'List at least "3" of the options that might hold between "{e1}" and "{e2}" or between "{e2}" and "{e1}" in the sentence. Be as permissive as possible when selecting the relations that might hold. Format your response as a python list of single letters without any additional things including your notes and explanations.'

    if model_family == Model_Family.LLAMA_31_8B:
        prompt += " Make sure that you format your response as a" \
                  " python list of single letters each enclosed by single quotes" + \
                  " without any notes and explanations."

    return prompt


def eval_reduce_label_space(model_family, client, model, tokenizer, dataset, n, experiment_name, open_source_api_mode=None):
    sleep = False
    save = False

    if model_family == Model_Family.OPENAI:
        pipe = client
    else:
        if open_source_api_mode is None:
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
            )
        else:
            pipe = client

    semeval_path = "../../data/semeval/"

    if dataset == "train.0.01":
        with open(f"{semeval_path}train.0.01.json", "r") as f:
            examples = json.load(f)

    elif dataset == "test_subset":
        with open(f"{semeval_path}test_subset.json", "r") as f:
            examples = json.load(f)

    else:
        raise Exception(f"Dataset {dataset} not recognized.")


    labels = []
    outputs = []
    wrong_label_reduction = []
    counter = 0
    predictions = ""
    corrects = 0
    list_lens = []
    reduced_options = {}

    for example in tqdm(examples, total=len(examples)):
        if sleep:
            time.sleep(1)

        prompt = get_prompt(example, n, experiment_name, model_family)

        response = get_completion(open_source_api_mode, model_family, prompt, pipe, model)
        response = trim_response(response).strip()

        relation_set = get_relation_set()
        example_id = example["id"]

        try:
            list_of_options = literal_eval(response)

            list_of_relations = []
            for option in list_of_options:
                relation = match_response_to_label(option, relation_set)
                list_of_relations.append(relation)

        except Exception as e:
            print("\nError:")
            print(str(e))
            print(f"id: {example_id}")
            print(f'context: {example["tagged_context"]}')
            print(f"response: {response}\n\n")

            try:
                print("Trying again...")

                if model_family == Model_Family.LLAMA_31_8B:
                    retry_prompt = prompt + " Make sure that you format your response as a" \
                                            " python list of single letters each enclosed by single quotes" + \
                                            " without any notes and explanations."
                else:
                    retry_prompt = prompt + " Make sure that you format your response as a" \
                                            " python list of single letters each enclosed by single quotes."

                retry_response = get_completion(open_source_api_mode, model_family, retry_prompt, pipe, model)
                retry_response = trim_response(retry_response).strip()

                list_of_options = literal_eval(retry_response)
                list_of_relations = []
                for option in list_of_options:
                    relation = match_response_to_label(option, relation_set)
                    list_of_relations.append(relation)

            except Exception as e_retry:
                print("\nError in retrying:")
                print(str(e_retry))
                print(f"retry_response: {retry_response}\n\n")
                wrong_label_reduction.append((example,None))
                continue

        label = example["label"]
        if label != "Other":
            label = label[:-7]

        if "Other" not in list_of_relations:
            list_of_relations.append("Other")

        reduced_options[example_id] = list_of_relations

        if label in list_of_relations:
            corrects += 1
        else:
            wrong_label_reduction.append((example,list_of_relations))

        list_len = len(list_of_relations)
        list_lens.append(list_len)

    if save:
        with open("experiments/wrong_label_reduction.json", "w") as f:
            json.dump(wrong_label_reduction, f, indent=4)

    reduced_options_filename = f"reduced_options_3_{dataset}_{experiment_name}.json"

    with open(f"json/{reduced_options_filename}", "w") as f:
        json.dump(reduced_options, f, indent=4)

    if dataset != "train.0.01":
        total = len(examples)
        acc = round(100 * corrects/total, 1)
        print(f"Accuracy (percentage of reduced examples containing gold labels): {acc}%")

        m = round(statistics.mean(list_lens), 1)
        std = round(statistics.stdev(list_lens), 1)
        print(f"Number of labels per example reduced to: {m} ± {std}")

