import json
import time
import sys
import os
import datetime

import openai
import backoff

from transformers import pipeline
import torch
from tqdm import tqdm
import numpy as np

from semeval_utils import get_relation_set, get_relation_set_extended, get_options, get_relation_set_string,\
                          get_definition, get_examples, Model_Family, get_label_id,\
                          precision_recall_fscore_exclude_nota, Open_Source_API_Mode


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_completion_openai(prompt, client, model, messages):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


def get_completion_open_source(prompt, pipe, messages):
    if messages is None:
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


def get_completion_open_source_api(prompt, client, model, messages):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0
    )

    return completion.choices[0].message.content


def get_completion_mistral_api(prompt, client, model, messages):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    chat_response = client.chat.complete(
        model= model,
        messages = messages,
        temperature=0,
        max_tokens=256,
    )

    return chat_response.choices[0].message.content


def get_completion(open_source_api_mode, model_family, prompt, pipe, model, messages=None):
    if model_family == Model_Family.OPENAI:
        return get_completion_openai(prompt, pipe, model, messages)

    elif (model_family == Model_Family.OPEN_SOURCE) or (model_family == Model_Family.LLAMA_31_8B):
        if open_source_api_mode is None:
            return get_completion_open_source(prompt, pipe, messages)

        elif open_source_api_mode == Open_Source_API_Mode.HUGGING_FACE:
            return get_completion_open_source_api(prompt, pipe, model, messages)

        elif open_source_api_mode == Open_Source_API_Mode.MISTRAL:
            return get_completion_mistral_api(prompt, pipe, model, messages)

        else:
            raise Exception(f"Open source API mode '{open_source_api_mode}' not supported.")

    else:
        raise Exception(f"Model Family '{model_family}' Not Supported.")


def get_completion_from_messages(open_source_api_mode, model_family, pipe, model, messages):
    return get_completion(open_source_api_mode, model_family, None, pipe, model, messages)


def get_clean_entity_type(entity_type):
    clean_entity_type = entity_type.replace("_", " ")
    return clean_entity_type


def trim_response(response):
    if response[:8] == "```json\n":
        return response[8:-4]
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


def get_prompt(example, n, relation_set, experiment_name):
    e1 = example["e1"]
    e2 = example["e2"]
    context = example["context"]
    example_id = example["id"]

    _, _, option_list_string = get_options(relation_set)
    relation_set_string = get_relation_set_string(relation_set)

    prompt = "I would like you to perform the task of relation extraction. " \
             "In this task, you are given a sentence and a pair of entities in the sentence. " \
             "Your job is to select the relation between the two entities from a predefined set of candidate relations.\n\n" \
    
    prompt += f"The predefined relations are {relation_set_string}.\n\n"

    prompt += 'The definition of each relation is as follows. Note that in relation examples or relation instances, X and Y are replaced with actual words.'

    for relation in relation_set:
        if relation == "Other":
            continue
        relation_label = relation

        filename = "json/relations_descriptions_complete_v1.json"
        relation_definition = get_definition(relation, filename)

        prompt += f'\n\n{relation_label}:\n{relation_definition}\n\n'

        if n > 0:
            if n == 1:
                prompt += f'Below is an example sentence for the "{relation_label}" relation:'
            else:
                prompt += f'Below are {n} example sentences for the "{relation_label}" relation:'

            file_name = f"json/relations_patterns_v4_{experiment_name}.json"
            patterns = get_examples(relation, file_name, n)
            for pattern in patterns:
                prompt += f"\n{pattern}"

    prompt += '\n\nIf none of the above relations holds between the two entities, you should output "Other".'

    with open("json/general_guidelines.json", "r") as f:
        general_guidelines = json.load(f)
    general_guidelines = general_guidelines.replace("<e1>","").replace("</e1>","").replace("<e2>","").replace("</e2>","")
    prompt += f'\n\n{general_guidelines}'

    prompt += f'\n\nNow, determine which option is the relation between "{e1}" and "{e2}" or between "{e2}" and "{e1}" in the following sentence.\n\n'

    prompt += f"Sentence: {context}\n\n"

    prompt += f"Options:\n{option_list_string}\n"

    prompt += f'Which option is the relation between "{e1}" and "{e2}" or between "{e2}" and "{e1}" in the above sentence? Respond with a single letter without any additional things including your notes and explanations.'

    return prompt


def generate_second_response(open_source_api_mode, first_prompt, first_response, second_prompt, model_family, pipe, model):
    messages = [{"role": "user", "content": first_prompt}]
    messages.append({'role':'assistant', 'content':f"{first_response}"})

    messages.append({"role": "user", "content": second_prompt})
    second_response = get_completion_from_messages(open_source_api_mode, model_family, pipe, model, messages)

    return second_response


def get_second_prompt(predicted_label, example):
    e1 = example["e1"]
    e2 = example["e2"]

    with open("json/relations_descriptions_simple_v0.json", "r") as f:
        relations_descriptions = json.load(f)
    rel_desc = relations_descriptions[predicted_label]
    
    option_string1 = rel_desc.replace("X", e1).replace("Y", e2)
    option_string2 = rel_desc.replace("X", e2).replace("Y", e1)
    option_strings = [option_string1, option_string2]

    _, _, option_list_string = get_options(option_strings)

    second_prompt = f'If the relation between "{e1}" and "{e2}" or between "{e2}" and "{e1}" is "{predicted_label}", then which one of the following options is correct?\n\n'

    second_prompt += f"Options:\n{option_list_string}\n"

    second_prompt += "Respond with a single letter without any additional things including your notes and explanations."

    return second_prompt, option_strings


def eval_final(model_family, client, model, tokenizer, dataset, n, experiment_name, open_source_api_mode=None):
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

    reduced_options_filename = f"reduced_options_3_{dataset}_{experiment_name}.json"

    with open(f"json/{reduced_options_filename}", "r") as f:
        reduced_options = json.load(f)


    labels = []
    outputs = []
    wrong_preds = []

    for example in tqdm(examples, total=len(examples)):
        if sleep:
            time.sleep(1)

        if example["id"] not in reduced_options:
            relation_set = list(get_relation_set())
        else:
            relation_set = reduced_options[example["id"]]

        prompt = get_prompt(example, n, relation_set, experiment_name)

        response = get_completion(open_source_api_mode, model_family, prompt, pipe, model)
        orig_response = response

        if model_family == Model_Family.OPENAI:
            response = trim_response(response).strip()
        else:
            response = response.strip()

        example_id = example["id"]
        label = example["label"]
        label_id = get_label_id(label)
        labels.append(label_id)

        try:
            predicted_label = match_response_to_label(response, relation_set)

        except:
            context = example["context"]
            relation_set_string = get_relation_set_string(relation_set)

            print("\nError:")
            print(f"id: {example_id}")
            print(f"context: {context}")
            print(f"relation set: {relation_set_string}")
            print(f"label: {label}")
            print(f"response: {response}\n")

            predicted_label = "Other"


        if predicted_label != "Other":
            second_prompt , option_strings = get_second_prompt(predicted_label, example)

            second_response = generate_second_response(open_source_api_mode, prompt, orig_response, second_prompt, model_family, pipe, model)
            orig_second_response = second_response

            if model_family == Model_Family.OPENAI:
                second_response = trim_response(second_response).strip()
            else:
                second_response = second_response.strip()

            try:
                if second_response in ["A", "A."]:
                    prediction = predicted_label + "(e1,e2)"
                elif second_response in ["B", "B."]:
                    prediction = predicted_label + "(e2,e1)"
                else:
                    raise Exception("Couldn't match the response.")

            except Exception as e:
                print("\nError:")
                print(str(e))
                print(f"id: {example_id}")
                print(f"second response: {orig_second_response}\n")
                
                try:
                    print("Trying again...")
                    retry_second_prompt = second_prompt + " There are two options: A or B?"

                    retry_second_response = generate_second_response(open_source_api_mode, prompt, orig_response, retry_second_prompt, model_family, pipe, model)
                    retry_second_response = trim_response(retry_second_response).strip()

                    if retry_second_response in ["A", "A."]:
                        prediction = predicted_label + "(e1,e2)"
                    elif retry_second_response in ["B", "B."]:
                        prediction = predicted_label + "(e2,e1)"
                    else:
                        raise Exception("Couldn't match the retried response.")

                except Exception as e_retry:
                    print("\nError in retrying:")
                    print(str(e_retry))
                    print(f"retry_second_response: {retry_second_response}\n\n")

                    prediction = predicted_label + "(e1,e2)"

        else:   
            prediction = predicted_label

        predicted_label_id = get_label_id(prediction)
        outputs.append(predicted_label_id)

        if predicted_label_id != label_id:
            relation_set_string = get_relation_set_string(relation_set)
            wrong_preds.append((example, prediction, relation_set_string))


    labels = np.array(labels)
    outputs = np.array(outputs)

    if save:
        np.save("experiments/labels.npy", labels)
        np.save("experiments/outputs.npy", outputs)
        with open("experiments/wrong_preds.json", "w") as f:
            json.dump(wrong_preds, f, indent=4)

    p, r, f1 = precision_recall_fscore_exclude_nota(labels, outputs)
    print(f"\np:{p} r:{r} f1:{f1}")

    return f1

