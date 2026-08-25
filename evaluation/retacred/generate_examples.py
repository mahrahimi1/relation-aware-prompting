import json
import time
import re

import openai
from openai import OpenAI
import backoff

from transformers import pipeline
import torch
from tqdm import tqdm

from retacred_utils import Model_Family, Open_Source_API_Mode


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
        max_new_tokens=1000,
        do_sample=False,
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
        max_tokens=1000,
        temperature=0
    )

    return completion.choices[0].message.content


def get_completion_mistral_api(prompt, client, model):
    messages = [{"role": "user", "content": prompt}]

    chat_response = client.chat.complete(
        model= model,
        messages = messages,
        temperature=0,
        max_tokens=1000,
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


def get_clean_relation_label(relation_label):
    clean_relation_label = relation_label.split(":")[1]
    clean_relation_label = clean_relation_label.replace("_", " ")
    clean_relation_label = clean_relation_label.replace("/", " or ")
    clean_relation_label = clean_relation_label.replace("stateorprovince", "state or province")
    return clean_relation_label


def get_clean_entity_type(entity_type):
    clean_entity_type = entity_type.replace("_", " ")
    return clean_entity_type


def generate_prompt_orig(relation_label, subj_type, obj_type, relation_description, n):
    relation_description = relation_description[0].lower() + relation_description[1:]

    if relation_label == "identity":
        relation_description = relation_description
    else:
        relation_description = "the tail entity is " + relation_description 

    prompt = f'Please give {n} examples of the relation "{relation_label}" between a {subj_type} (called head entity) and a {obj_type} (called tail entity) ' \
             f'using a single subject-verb-object structure containing the head entity and the tail entity.\n' \
             f'Note that {relation_description}\n' \
             f'Do not overfit the pattern of the above definition. ' \
             f'Try as many different relation patterns or relation expressions as possible.\n' \
             f'Prefix the head entity with tag <ENT0> and suffix it with tag </ENT0>. Also, ' \
             f'prefix the tail entity with tag <ENT1> and suffix it with tag </ENT1>.\n' \
             f'Produce your response as a list of strings in a json list object.'

    return prompt


def generate_prompt_llama31_8b(relation_label, subj_type, obj_type, relation_description, n):
    relation_description = relation_description[0].lower() + relation_description[1:]

    if relation_label == "identity":
        relation_description = relation_description
    else:
        relation_description = "the tail entity is " + relation_description

    prompt = f'Please give {n} example sentences of the relation "{relation_label}" between an actual entity mention of a ' +\
             f'{subj_type} (called head entity) and an actual entity mention of a {obj_type} (called tail entity) ' \
             f'using a single subject-verb-object structure containing the head entity and the tail entity.\n' \
             f'Note that {relation_description}\n' \
             f'Do not overfit the pattern of the above definition. ' \
             f'Try as many different relation patterns or relation expressions as possible.\n' \
             f'Prefix the head entity with tag <ENT0> and suffix it with tag </ENT0>. Also, ' \
             f'prefix the tail entity with tag <ENT1> and suffix it with tag </ENT1>.\n' \
             f'Produce your response as a list of strings in a json list object without any notes and explanations.'

    return prompt


def generate_prompt(model_family, relation_label, subj_type, obj_type, relation_description, n):
    if model_family == Model_Family.LLAMA_31_8B:
        return generate_prompt_llama31_8b(relation_label, subj_type, obj_type, relation_description, n)

    else:
        return generate_prompt_orig(relation_label, subj_type, obj_type, relation_description, n)


def process_reposnse(response, subj_type, obj_type):
    not_replaced_types = ["title"]

    subj_types = [subj_type, subj_type.lower(), subj_type.upper(), subj_type.capitalize(), subj_type.title()]
    subj_types = list(set(subj_types))

    for subj_type in subj_types:
        if subj_type.lower() not in not_replaced_types:
            response = response.replace(f"{subj_type} X", "X")

    obj_types = [obj_type, obj_type.lower(), obj_type.upper(), obj_type.capitalize(), obj_type.title()]
    obj_types = list(set(obj_types))

    for obj_type in obj_types:
        if obj_type.lower() not in not_replaced_types:
            response = response.replace(f"{obj_type} Y", "Y")

    return response


def check(patterns, n):
    if (type(patterns) != list) and (type(patterns) != dict):
        print("check pattern failed")
        return False

    if len(patterns) != n:
        print("check pattern failed")
        return False

    return True


def check_ent0_ent1(patterns):
    for pattern in patterns:
        p0 = r"<ENT0>.+?</ENT0>"
        match = re.search(p0, pattern)
        if match is None:
            print("check ENT0 ENT1 failed.")
            return False
        else:
            updated_pattern = re.sub(p0, "", pattern, count=1)

        p1 = r"<ENT1>.+?</ENT1>"
        match = re.search(p1, updated_pattern)
        if match is None:
            print("check ENT0 ENT1 failed.")
            return False

    return True


def check_ent0_ent1_permissive(patterns):
    for pattern in patterns:
        p1 = r"<ENT\d>.+?</ENT\d>"
        match = re.search(p1, pattern)
        if match is None:
            print("check permissive ENT0 ENT1 failed.")
            return False
        else:
            updated_pattern = re.sub(p1, "", pattern, count=1)

        p0 = r"<ENT\d>.+?</ENT\d>"
        match = re.search(p0, updated_pattern)
        if match is None:
            print("check permissive ENT0 ENT1 failed.")
            return False

    return True


def convert_dict_to_list(patterns):
    if type(patterns) == dict:
        l = []
        for _ , pattern in patterns.items():
            l.append(pattern)
        patterns = l

    return patterns


def check_XY(patterns):
    for pattern in patterns:
        if "X" not in pattern:
            print("checkXY pattern failed")
            return False

        if "Y" not in pattern:
            print("checkXY pattern failed")
            return False

    return True


def trim_json(response):
    if response[:7] == "```json":
        return response[7:-3]

    elif response[:3] == "```":
        return response[3:-3].strip()

    else:
        return response


def generate_examples(model_family, client, model, tokenizer, experiment_name, open_source_api_mode=None):
    n = 10 #number of generated patterns per relation

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

    with open("json/relations_types_for_patterns.json", "r") as relations_types_file:
        relations_types = json.load(relations_types_file)

    with open("json/relations_descriptions.json", mode='r', encoding="utf_8") as file:
        relations_descriptions = json.load(file)

    relations_patterns = {}

    for relation, types in tqdm(relations_types.items(), total=len(relations_types)):
        relation_label = get_clean_relation_label(relation)
        relation_description = relations_descriptions[relation]

        relations_patterns[relation] = []

        for subj_type in types["subj types"]:
            for obj_type in types["obj types"]:
                subj_type = get_clean_entity_type(subj_type).lower()
                obj_type = get_clean_entity_type(obj_type).lower()
                
                prompt = generate_prompt(model_family, relation_label, subj_type, obj_type, relation_description, n)
                orig_prompt = prompt

                counter = 0

                error_correction_prompt = ""
                ent_correction_prompt = ""
                while (True):
                    counter += 1
                    if counter > 10:
                        exit()

                    response = get_completion(open_source_api_mode,
                                              model_family, 
                                              prompt + error_correction_prompt + ent_correction_prompt, 
                                              pipe, 
                                              model)
                    response = trim_json(response)
                    response = process_reposnse(response, subj_type, obj_type)

                    try:
                        patterns = json.loads(response)
                    except:
                        print("\nError!\n")
                        patterns = None

                    print(response,"\n")

                    if model_family == Model_Family.OPENAI:
                        if check(patterns, n) == False:
                            continue

                    else:
                        if check(patterns, n) == False:
                            error_correction_prompt = " Only respond with a json list. Do not include notes and explanations in your response."
                            continue

                        if model_family == Model_Family.LLAMA_31_8B:
                            if check_ent0_ent1(patterns) == False:
                                ent_correction_prompt = f" In your sentences, make sure the head entity is an actual entity " + \
                                                f"mention of a {subj_type} and the tail entity is an actual entity " + \
                                                f"mention of a {obj_type}. Prefix the head entity with tag <ENT0> and " + \
                                                f"suffix it with tag </ENT0>. Also, prefix the tail entity with tag " + \
                                                f"<ENT1> and suffix it with tag </ENT1>."


                                if counter == 10:
                                    if check_ent0_ent1_permissive(patterns) == False:
                                        continue
                                    else:
                                        print("\npermissive check ent0 ent1 passed.\n")
                                else:
                                    continue

                    patterns = convert_dict_to_list(patterns)
                    break

                relations_patterns[relation].append( (subj_type, obj_type, patterns) )

    with open(f"json/relations_patterns_v4_{experiment_name}.json", "w") as outfile:
        json.dump(relations_patterns, outfile, indent=4)

