import json
import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
# import tiktoken
import requests
import yaml
from tqdm import tqdm

import dataset_analysis
import method_fvl
import search_engine_call as se_call
import llm_call
from utility import load_se_qa_dict, load_llm_prompt_answer_dict, get_current_upper_bound, get_llm_model_name

## for initializing baichuan2-server
def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    env_var_name, default_value = re.match(r"\$\{(.+?):(.+?)\}", value).groups()
    return os.environ.get(env_var_name, default_value)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Register the constructor with PyYAML
yaml.SafeLoader.add_implicit_resolver('!env_var', re.compile(r"\$\{(.+?):(.+?)\}"), None)
yaml.SafeLoader.add_constructor('!env_var', env_var_constructor)

model_path = '/root/llama_models/Baichuan2-13B-Chat'

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,
    trust_remote_code=True
)


def retrieve_webpage_content(se_qa_dict: dict, link: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                                '(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'}
        response = requests.get(url=link, headers=headers)
        status_code = response.status_code
        if status_code == 404:
            content = '404_error'
        else:
            # retrieve top-1 results from search engine
            if link not in se_qa_dict:
                try:
                    se_answer_list = se_call.get_response_from_google_serper(link, 1, 'english')
                    se_qa_dict[link] = se_answer_list
                    if len(se_answer_list) > 0:
                        title = se_answer_list[0]['title']
                        snippet = se_answer_list[0]['snippet']
                        content = title + '; ' + snippet
                    else:
                        content = ''
                except:
                    content = 'SE_ServiceUnavailableError'
            else:
                se_answer_list = se_qa_dict[link]
                title = se_answer_list[0]['title']
                snippet = se_answer_list[0]['snippet']
                content = title + '; ' + snippet
    except:
        # print('Connection Error! ' + link)
        content = 'connection_error'

    return se_qa_dict, content


def generate_evidence_link_relatedness_prompt(evidence: str, webpage_content: str):
    base = "您好，ChatGPT，我需要您的帮助，请判断两个文本evidence与content之间的相关性并提供解释。" \
           "如果相关，relatedness为1，不相关则为0。" \
           "以下是样例供参考。" \
           "<evidence>:<Locus Robotics的总部位于美国麻省安多弗伯特路100号>" \
           "<content>:<Locus Robotics; Locus Robotics, an innovative robotic process automation company offers automated " \
           "warehouse robots that increase productivity, order accuracy and more.>" \
           "<relatedness>:<1>" \
           "<explain>:<两个文本之间的相关性很高。evidence提供了关于Locus Robotics总部位置的具体信息，" \
           "而content提供了关于Locus Robotics的详细描述，包括其提供的自动化仓库机器人能够提高生产力和订单准确性等信息。" \
           "因此，evidence和content之间存在明显的相关性。>" \
           "以下是问题："
    evidence_part = "<evidence>:<" + evidence + ">"
    content_part = "<content>:<" + webpage_content + ">"
    prompt = base + evidence_part + content_part

    return prompt


def check_evidence_link_relatedness_by_llm(llm_pa_dict: dict, evidence: str, webpage_content: str,
                                           llm_type: str, llm_model_name: str, llm_temperature: str):
    elr_prompt = generate_evidence_link_relatedness_prompt(evidence, webpage_content)
    if elr_prompt in llm_pa_dict:
        response = llm_pa_dict[elr_prompt]
    else:
        try:
            if llm_type == 'baichuan2-server':
                response = llm_call.get_response_from_baichuan2_server(
                    model, tokenizer, elr_prompt, llm_temperature
                )
            else:
                response = llm_call.get_response_from_llm(llm_type, elr_prompt, llm_model_name, llm_temperature)
            llm_pa_dict[elr_prompt] = response
        except:
            print('LLM_ServiceUnavailableError: evidence_ ' + '{}'.format(evidence))
            response = 'LLM_ServiceUnavailableError'

    return llm_pa_dict, response


def check_fact_by_llm(fact_id: int, llm_pa_dict: dict, head: str, relation: str, tail: str,
                      llm_type: str, llm_model_name: str, llm_temperature: str):
    validation_prompt = method_fvl.generate_fvl_prompt(head, relation, tail)
    if validation_prompt in llm_pa_dict:
        response = llm_pa_dict[validation_prompt]
        slot_list = method_fvl.parse_llm_response(response)
        veracity = slot_list[1]
        evidence = slot_list[3]
        link = slot_list[5]
    else:
        try:
            if llm_type == 'baichuan2-server':
                response = llm_call.get_response_from_baichuan2_server(
                    model, tokenizer, validation_prompt, llm_temperature
                )
            else:
                response = llm_call.get_response_from_llm(llm_type, validation_prompt, llm_model_name, llm_temperature)
        except:
            print('LLM ServiceUnavailableError: fact-id_' + '{}'.format(fact_id) +
                  '_llm_type_' + llm_type)
            response = 'LLM_ServiceUnavailableError'
        # 从回答<veracity>:<真>, <evidence>:<Agility Robotics developed Cassie>,
        # <link>:<https://spectrum.ieee.org/agility-robotics-introduces-cassie-a-dynamic-and-talented-robot-delivery-ostrich>
        # 中提取veracity, evidence, link
        if response == 'LLM_ServiceUnavailableError':
            veracity = evidence = link = "llm_no_response_error"
        else:
            slot_list = method_fvl.parse_llm_response(response)
            if len(slot_list) == 6:
                veracity = slot_list[1]
                evidence = slot_list[3]
                link = slot_list[5]
                llm_pa_dict[validation_prompt] = response
            else:
                veracity = evidence = link = "llm_response_format_error"

    return llm_pa_dict, veracity, evidence, link, validation_prompt


def fvlc_validate_single_fact(validation_dict: dict, se_qa_dict: dict, llm_pa_dict: dict, fact_id: int,
                              head: str, relation: str, tail: str,
                              valid_llm_type: str, valid_llm_model_name: str, valid_llm_temp: str,
                              check_llm_type: str, check_llm_model_name: str, check_llm_temp: str,
                              head_mid: str, relation_mid: str):
    llm_pa_dict, veracity, evidence, link, fv_prompt = check_fact_by_llm(
        fact_id, llm_pa_dict, head, relation, tail,
        valid_llm_type, valid_llm_model_name, valid_llm_temp
    )

    if link.startswith('https') or link.startswith('http'):
        se_qa_dict, webpage_content = retrieve_webpage_content(se_qa_dict, link)
    else:
        # for example, 无、llm_response_format_error、N/A、llm_no_response_error
        webpage_content = 'link_not_valid'

    error_message_list = ['404_error', 'connection_error', 'link_not_valid']

    if webpage_content in error_message_list:
        relatedness = -1
        explain = 'link or connection error'
    elif webpage_content == 'SE_ServiceUnavailableError':
        relatedness = -2
        explain = 'search engine error'
        print(webpage_content + ': ' + link)
    else:
        llm_pa_dict, llm_response = check_evidence_link_relatedness_by_llm(
            llm_pa_dict, evidence, webpage_content,
            check_llm_type, check_llm_model_name, check_llm_temp
        )
        slot_list = method_fvl.parse_llm_response(llm_response)
        if len(slot_list) != 4:
            relatedness = -3
            explain = 'llm response format error'
        else:
            if slot_list[1] in ['-3', '-2', '-1', '0', '1']:
                relatedness = int(slot_list[1])
            else:
                relatedness = -4
            explain = slot_list[3]

    validation_dict[fact_id] = {'fact_id': fact_id, 'head': head, 'relation': relation, 'tail': tail,
                                'veracity': veracity, 'evidence': evidence, 'link': link,
                                'webpage_content': webpage_content,
                                'evidence_link_relatedness': relatedness, 'relatedness_explain': explain,
                                'fv_prompt': fv_prompt, 'head_mid': head_mid, 'relation_mid': relation_mid}

    return validation_dict, se_qa_dict, llm_pa_dict


def fvlc_validate_facts(input_file_path: str, output_directory: str, index_lower_bound: int,
                        index_upper_bound: int, dataset_name: str,
                        valid_llm_type: str, valid_llm_model_name: str, valid_llm_temp: str,
                        check_llm_type: str, check_llm_model_name: str, check_llm_temp: str):
    if dataset_name == 'robot':
        input_data = method_fvl.read_excel_to_dataframe(input_file_path)
    elif dataset_name == 'duie':
        input_data = pd.read_json(input_file_path, lines=True)
    elif dataset_name == 'okele':
        temp_df = pd.read_json(input_file_path, lines=True)
        input_data = []
        for index, value in temp_df.iterrows():
            for i, v in value.items():
                input_data.append(v)
    else:
        raise RuntimeError('Unknown dataset name: ' + dataset_name)

    # load search engine qa log if exists
    se_qa_file_path = os.path.join(os.getcwd(), 'data', 'se_llm_cache', 'se_qa_dict.json')
    se_qa_dict = load_se_qa_dict(se_qa_file_path)
    # load llm prompt answer log if exists
    if valid_llm_type in ['gpt3.5', 'gpt4']:
        llm_pa_file_path = os.path.join(
            os.getcwd(), 'data', 'se_llm_cache',
            'llm_pa_dict_' + valid_llm_type + '_' + valid_llm_model_name + '_' + valid_llm_temp + '.json'
        )
    else:
        llm_pa_file_path = os.path.join(
            os.getcwd(), 'data', 'se_llm_cache',
            'llm_pa_dict_' + valid_llm_type + '_' + valid_llm_temp + '.json'
        )
    llm_pa_dict = load_llm_prompt_answer_dict(llm_pa_file_path)

    validation_dict = {}
    input_data_selected = input_data[index_lower_bound: index_upper_bound]
    last_veracity = ''
    last_elr = -1000
    if dataset_name == 'robot':
        for index, row in tqdm(input_data_selected.iterrows(), total=input_data_selected.shape[0], desc='Processing'):
            time.sleep(1)
            fact_id = index
            head = row['head']
            relation = row['relation']
            tail = row['tail']
            validation_dict, se_qa_dict, llm_pa_dict = fvlc_validate_single_fact(
                validation_dict, se_qa_dict, llm_pa_dict, fact_id, head, relation, tail,
                valid_llm_type, valid_llm_model_name, valid_llm_temp,
                check_llm_type, check_llm_model_name, check_llm_temp, '', ''
            )
            veracity = validation_dict[fact_id]['veracity']
            evidence_link_relatedness = validation_dict[fact_id]['evidence_link_relatedness']
            if veracity == 'llm_no_response_error' or evidence_link_relatedness == -2:
                index_upper_bound = fact_id
                break
    elif dataset_name == 'duie':
        for index, row in tqdm(input_data_selected.iterrows(), total=input_data_selected.shape[0], desc='Processing'):
            if last_veracity == 'llm_no_response_error' or last_elr == -2:
                break
            time.sleep(1)
            # in duie dataset, fact_id equals to golden context_id (index) multiplied by 100 plus spo numbers
            fact_id = index * 100
            spo_list = row['spo_list']
            for spo in spo_list:
                head = spo['subject']
                relation = spo['predicate']
                object_dict = spo['object']
                tail = object_dict['@value']
                fact_id += 1
                validation_dict, se_qa_dict, llm_pa_dict = fvlc_validate_single_fact(
                    validation_dict, se_qa_dict, llm_pa_dict, fact_id, head, relation, tail,
                    valid_llm_type, valid_llm_model_name, valid_llm_temp,
                    check_llm_type, check_llm_model_name, check_llm_temp, '', ''
                )
                veracity = validation_dict[fact_id]['veracity']
                evidence_link_relatedness = validation_dict[fact_id]['evidence_link_relatedness']
                if veracity == 'llm_no_response_error' or evidence_link_relatedness == -2:
                    last_veracity = veracity
                    last_elr = evidence_link_relatedness
                    index_upper_bound = index
                    break
    elif dataset_name == 'okele':
        for index in tqdm(range(len(input_data_selected))):
            if last_veracity == 'llm_no_response_error' or last_elr == -2:
                break
            time.sleep(0.1)
            fact_id = (index + index_lower_bound) * 100
            row = input_data_selected[index]
            entity = row['entity']
            head_label = entity['lable']
            head_mid = entity['mid']
            relation = row['relation']
            relation_label = relation['lable']
            relation_mid = relation['mid']
            values = row['values']

            for value in values:
                tail = value['value']
                fact_id += 1
                validation_dict, se_qa_dict, llm_pa_dict = fvlc_validate_single_fact(
                    validation_dict, se_qa_dict, llm_pa_dict, fact_id, head_label, relation_label, tail,
                    valid_llm_type, valid_llm_model_name, valid_llm_temp,
                    check_llm_type, check_llm_model_name, check_llm_temp,
                    head_mid, relation_mid
                )
                veracity = validation_dict[fact_id]['veracity']
                evidence_link_relatedness = validation_dict[fact_id]['evidence_link_relatedness']
                if veracity == 'llm_no_response_error' or evidence_link_relatedness == -2:
                    last_veracity = veracity
                    last_elr = evidence_link_relatedness
                    index_upper_bound = index + index_lower_bound
                    break
    else:
        raise RuntimeError('Unknown dataset name: ' + dataset_name)

    # save validation dict as excel format, if it exists
    if len(validation_dict) > 0 and index_lower_bound != index_upper_bound:
        output_excel_path = os.path.join(
            output_directory,
            'fvlc_valid_' + valid_llm_type + '_' + valid_llm_temp +
            '_check_' + check_llm_type + '_' + check_llm_temp +
            '_from_' + '{}'.format(index_lower_bound) + '_to_' + '{}'.format(index_upper_bound) + '.xlsx'
        )
        dataset_analysis.save_to_excel(validation_dict, output_excel_path)
        # save search engine qa log to reduce the number of api calls
        with open(se_qa_file_path, 'w', encoding='utf-8') as f:
            json.dump(se_qa_dict, f)
        # save llm prompt answer log to reduce the number of api calls
        with open(llm_pa_file_path, 'w', encoding='utf-8') as f:
            json.dump(llm_pa_dict, f)

    return 0


def get_input_file_and_output_directory(dataset_name: str):
    if dataset_name == 'robot':
        input_file_path = os.path.join(os.getcwd(), 'data', 'Robot', 'original_knowledge.xlsx')
        output_directory = os.path.join(os.getcwd(), 'analysis', 'fact_validation', 'FVLC', 'Robot')
    elif dataset_name == 'duie':
        input_file_path = os.path.join(os.getcwd(), 'data', 'DuIE', 'duie_train.json')
        output_directory = os.path.join(os.getcwd(), 'analysis', 'fact_validation', 'FVLC', 'DuIE')
    elif dataset_name == 'okele':
        input_file_path = os.path.join(os.getcwd(), 'data', 'OKELE', 'processed', 'real-world.verification.okele.json')
        output_directory = os.path.join(os.getcwd(), 'analysis', 'fact_validation', 'FVLC', 'OKELE')
    else:
        raise Exception('Unknown dataset: %s' % dataset_name)

    # if input file does not exist, raise error
    if not os.path.exists(input_file_path):
        raise Exception('Input file does not exist: %s' % input_file_path)
    # if output directory does not exist, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return input_file_path, output_directory


def fvlc_validate_facts_recursively(dataset_name: str, index_lower_bound: int, index_upper_bound: int,
                                    valid_llm_type: str, valid_llm_model_name: str, valid_llm_temp: str,
                                    check_llm_type: str, check_llm_model_name: str, check_llm_temp: str):
    input_file_path, output_directory = get_input_file_and_output_directory(dataset_name)

    method = 'fvlc'
    cur_upper = get_current_upper_bound(output_directory, method,
                                        valid_llm_type, valid_llm_temp, 0, check_llm_type, check_llm_temp)
    while cur_upper < index_upper_bound:
        if cur_upper > index_lower_bound:
            fvlc_validate_facts(input_file_path, output_directory,
                                cur_upper, index_upper_bound, dataset_name,
                                valid_llm_type, valid_llm_model_name, valid_llm_temp,
                                check_llm_type, check_llm_model_name, check_llm_temp)
        else:
            fvlc_validate_facts(input_file_path, output_directory,
                                index_lower_bound, index_upper_bound, dataset_name,
                                valid_llm_type, valid_llm_model_name, valid_llm_temp,
                                check_llm_type, check_llm_model_name, check_llm_temp)
        cur_upper = get_current_upper_bound(output_directory, method,
                                            valid_llm_type, valid_llm_temp, 0, check_llm_type, check_llm_temp)
        time.sleep(10)

    return 0


if __name__ == '__main__':
    index_lower_bound = 0
    # 3098 for robot, 3000 for duie, 3060 for okele
    # index_upper_bound = 100
    # [gpt3.5, gpt4, baichuan2, chatglm-chat6b]

    dataset_name_list = ['duie', 'robot', 'okele']
    for dataset_name in dataset_name_list:
        if dataset_name == 'duie':
            index_upper_bound = 3000
        elif dataset_name == 'robot':
            index_upper_bound = 3098
        elif dataset_name == 'okele':
            index_upper_bound = 3060
        else:
            raise RuntimeError('Unknown dataset name: ' + dataset_name)
        # index_upper_bound = 100

        valid_llm_type_list = ['gpt3.5']
        for valid_llm_type in valid_llm_type_list:
            valid_llm_model_name = get_llm_model_name(valid_llm_type)
            # valid_llm_temp_list = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
            valid_llm_temp_list = ['0']
            for valid_llm_temp in valid_llm_temp_list:
                check_llm_type_list = ['gpt3.5']
                for check_llm_type in check_llm_type_list:
                    # if valid_llm_temp in ['chatglm-chat6b', 'gpt3.5']:
                    #     check_llm_type = 'gpt3.5'
                    # else:
                    #     check_llm_type = 'gpt4'
                    check_llm_model_name = get_llm_model_name(check_llm_type)
                    check_llm_temp = '0'
                    print(
                        'Start processing ' + dataset_name + '_fvlc_valid_' + valid_llm_type + '_' + valid_llm_temp +
                        '_check_' + check_llm_type + '_' + check_llm_temp + '_from_' + str(index_lower_bound) +
                        '_to_' + str(index_upper_bound))
                    fvlc_validate_facts_recursively(dataset_name, index_lower_bound, index_upper_bound,
                                                    valid_llm_type, valid_llm_model_name, valid_llm_temp,
                                                    check_llm_type, check_llm_model_name, check_llm_temp)
