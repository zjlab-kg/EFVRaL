import csv
import json
import os
import re
import time
import pandas as pd
from tqdm import tqdm
import search_engine_call as se_call
import llm_call
import dataset_analysis as da

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from utility import load_se_qa_dict, load_llm_prompt_answer_dict, get_current_upper_bound, \
    is_evidence_relevant, get_llm_model_name


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


def generate_prompt_for_validation(head: str, relation: str, tail: str, context: str):
    base = "您好，ChatGPT，我需要您的帮助，根据一段文字描述，且仅限于这段描述，述验证一个事实的真伪。" \
           "请按照[真伪；解释]的形式回答并只返回最有把握的一个。" \
           "真伪可以是“真”或者“非真”。以下是样例供参考：" \
           "文字描述：中国工程院院士王坚任之江实验室主任 - 新闻- 科学网；公开信息显示，王坚生于1962年10月，" \
           "1990年毕业于杭州大学心理系，获工程心理学博士学位。他曾任杭州大学心理学系主任，浙江大学理学院副院长，微软亚洲..." \
           "事实：(之江实验室，主任，王坚）" \
           "回答：[真；中国工程院院士王坚任之江实验室主任]" \
           "文字描述：看起来回报率不错。 改变几个月后，微软宣布注资10亿美元。 正如我们在GPT - 3和Codex上看到的那样，OpenAI" \
           "与微软的合作伙伴关系意味着允许后者将部分技术商业化" \
           "事实：（OpenAI，属于，微软）" \
           "回答：[非真；文中未提到OpenAI属于微软的任何直接证据]" \
           "请验证如下事实："

    text = "文字描述：" + context
    fact = "事实：（" + head + "，" + relation + "，" + tail + "）"

    prompt_for_validation = base + text + fact

    return prompt_for_validation


def select_top_k_se_answer(head: str, tail: str, se_answer_list_top_20: list, search_k: int):
    top_k_se_answer_list = []
    for se_answer in se_answer_list_top_20:
        title = se_answer['title']
        snippet = se_answer['snippet']
        context = title + '; ' + snippet
        if is_evidence_relevant(context, head, tail):
            top_k_se_answer_list.append(se_answer)
            if len(top_k_se_answer_list) >= search_k:
                break

    return top_k_se_answer_list


def validate_single_fact(validation_dict: dict, se_qa_dict: dict, llm_pa_dict: dict,
                         fact_id: int, text: str, head: str, relation: str, tail: str,
                         se_name: str, search_k: int, llm_type: str, llm_model_name: str, llm_temperature: str,
                         dataset_name: str, head_mid: str, relation_mid: str):
    if len(validation_dict) == 0:
        validation_id = 0
        se_call_count = 0
        llm_call_count = 0
    else:
        validation_id = len(validation_dict) - 1
        se_call_count = validation_dict[validation_id]['se_call_count']
        llm_call_count = validation_dict[validation_id]['llm_call_count']

    se_question = head + ' ' + relation + ' ' + tail
    if dataset_name == 'okele':
        language = 'english'
    else:
        language = 'chinese'

    if se_name == 'google':
        # retrieve top-20 results from search engine
        if se_question not in se_qa_dict:
            try:
                se_answer_list_top_20 = se_call.get_response_from_google_serper(se_question, 20, language)
                se_qa_dict[se_question] = se_answer_list_top_20
            except:
                print('SE_ServiceUnavailableError: ' + se_question)
                validation_dict[validation_id] = {'context': 'SE_ServiceUnavailableError', 'question': 'N/A',
                                                  'answer': 'N/A',
                                                  'type': 'N/A', 'origin': 'N/A', 'prompt': 'N/A',
                                                  'context_origin': 'N/A', 'fact_id': fact_id,
                                                  'validation_id': validation_id,
                                                  'text': text, 'se_call_count': se_call_count,
                                                  'llm_call_count': llm_call_count,
                                                  'head_mid': 'N/A', 'relation_mid': 'N/A', 'tail': tail
                                                  }
                return validation_dict, se_qa_dict, llm_pa_dict
            se_call_count += 1
            # print('se_call_count: ' + '{}'.format(se_call_count))
        else:
            se_answer_list_top_20 = se_qa_dict[se_question]
        # select top search_k se answers
        se_answer_list = select_top_k_se_answer(head, tail, se_answer_list_top_20, search_k)
    else:
        raise RuntimeError('Search engine name ' + se_name + ' not supported: ' + se_name)

    type = "自由文本"
    origin = "ChatGPT"
    for se_answer in se_answer_list:
        annotation_question = "（" + head + "，" + relation + "，" + tail + "），这个事实是真的吗？"
        title = se_answer['title']
        snippet = se_answer['snippet']
        link = se_answer['link']
        context = title + '; ' + snippet
        promt_for_validation = generate_prompt_for_validation(head, relation, tail, context)
        if promt_for_validation in llm_pa_dict:
            response = llm_pa_dict[promt_for_validation]
        else:
            try:
                if llm_type == 'baichuan2-server':
                    response_origin = llm_call.get_response_from_baichuan2_server(
                        model, tokenizer, promt_for_validation, llm_temperature
                    )
                else:
                    response_origin = llm_call.get_response_from_llm(
                        llm_type, promt_for_validation, llm_model_name, llm_temperature)
                # 从回答：[真；邬江兴于1995年获得何梁何利科学技术进步奖]中截取真；邬江兴于1995年获得何梁何利科学技术进步奖
                start = response_origin.find('[')
                end = response_origin.find(']')
                response_cut = response_origin[start + 1:end]
                response = response_cut.replace(' ', '')
                llm_pa_dict[promt_for_validation] = response
            except:
                print('LLM_ServiceUnavailableError: fact_id ' + '{}'.format(fact_id))
                response = 'LLM_ServiceUnavailableError'
                validation_dict[validation_id] = {'context': context, 'question': annotation_question,
                                                  'answer': response,
                                                  'type': type, 'origin': origin, 'prompt': promt_for_validation,
                                                  'context_origin': link, 'fact_id': fact_id,
                                                  'validation_id': validation_id,
                                                  'text': text, 'se_call_count': se_call_count,
                                                  'llm_call_count': llm_call_count,
                                                  'head_mid': head_mid, 'relation_mid': relation_mid, 'tail': tail}
                return validation_dict, se_qa_dict, llm_pa_dict
            llm_call_count += 1
            # print('llm_call_count: ' + '{}'.format(llm_call_count))
        if len(validation_dict) > 0:
            validation_id += 1

        validation_dict[validation_id] = {'context': context, 'question': annotation_question, 'answer': response,
                                          'type': type, 'origin': origin, 'prompt': promt_for_validation,
                                          'context_origin': link, 'fact_id': fact_id, 'validation_id': validation_id,
                                          'text': text, 'se_call_count': se_call_count,
                                          'llm_call_count': llm_call_count,
                                          'head_mid': head_mid, 'relation_mid': relation_mid, 'tail': tail}

    return validation_dict, se_qa_dict, llm_pa_dict


def save_to_csv(data_dict: dict, target: str):
    with open(target, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['context', 'question', 'answer', 'type', 'origin', 'prompt', 'context_origin'])
        for index in range(len(data_dict)):
            writer.writerow(data_dict[index].values())

    print('File generated successfully...' + target)


def get_input_file_and_output_directory(dataset: str):
    if dataset == 'robot':
        input_file_path = os.path.join(os.getcwd(), 'data', 'Robot', 'original_knowledge.xlsx')
        output_directory = os.path.join(os.getcwd(), 'analysis', 'fact_validation', 'FVReL', 'Robot')
    elif dataset == 'duie':
        input_file_path = os.path.join(os.getcwd(), 'data', 'DuIE', 'duie_train.json')
        output_directory = os.path.join(os.getcwd(), 'analysis', 'fact_validation', 'FVReL', 'DuIE')
    elif dataset == 'okele':
        input_file_path = os.path.join(os.getcwd(), 'data', 'OKELE', 'processed', 'real-world.verification.okele.json')
        output_directory = os.path.join(os.getcwd(), 'analysis', 'fact_validation', 'FVReL', 'OKELE')
    else:
        raise Exception('Unknown dataset: %s' % dataset)

    # if input file does not exist, raise error
    if not os.path.exists(input_file_path):
        raise Exception('Input file does not exist: %s' % input_file_path)
    # if output directory does not exist, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return input_file_path, output_directory


def validate_facts(dataset_name: str, index_lower_bound: int, index_upper_bound: int, se_name: str, search_k: int,
                   llm_type: str, llm_model_name: str, llm_temperature: str):
    input_file_path, output_directory = get_input_file_and_output_directory(dataset_name)
    input_file_suffix = os.path.splitext(input_file_path)[-1]
    if input_file_suffix == '.json':
        if dataset_name == 'okele':
            temp_df = pd.read_json(input_file_path, lines=True)
            input_data = []
            for index, value in temp_df.iterrows():
                for i, v in value.items():
                    input_data.append(v)
        else:
            input_data = pd.read_json(input_file_path, lines=True)
    elif input_file_suffix == '.xlsx':
        input_data = pd.read_excel(input_file_path)
    else:
        raise RuntimeError('Input file suffix Error: ' + input_file_suffix)

    # load search engine qa log if exists
    se_qa_file_path = os.path.join(os.getcwd(), 'data', 'se_llm_cache', 'se_qa_dict.json')
    se_qa_dict = load_se_qa_dict(se_qa_file_path)
    # load llm prompt answer log if exist
    if llm_type in ['gpt3.5', 'gpt4']:
        llm_pa_file_path = os.path.join(
            os.getcwd(), 'data', 'se_llm_cache',
            'llm_pa_dict_' + llm_type + '_' + llm_model_name + '_' + llm_temperature + '.json'
        )
    else:
        llm_pa_file_path = os.path.join(
            os.getcwd(), 'data', 'se_llm_cache',
            'llm_pa_dict_' + llm_type + '_' + llm_temperature + '.json'
        )

    llm_pa_dict = load_llm_prompt_answer_dict(llm_pa_file_path)

    validation_dict = {}

    last_answer = ''
    last_context = ''
    input_data_selected = input_data[index_lower_bound: index_upper_bound]
    if dataset_name in ['duie', 'robot']:
        for index, row in tqdm(input_data_selected.iterrows(), total=input_data_selected.shape[0], desc='Processing'):
            time.sleep(0.1)
            if dataset_name == 'duie':
                if last_answer == 'LLM_ServiceUnavailableError' or last_context == 'SE_ServiceUnavailableError':
                    break
                # in duie dataset, fact_id equals to golden context_id (index) multiplied by 100 plus spo numbers
                fact_id = index * 100
                text = row['text']
                spo_list = row['spo_list']
                for spo in spo_list:
                    head = spo['subject']
                    relation = spo['predicate']
                    object_dict = spo['object']
                    tail = object_dict['@value']
                    fact_id += 1
                    validation_dict, se_qa_dict, llm_pa_dict = validate_single_fact(
                        validation_dict, se_qa_dict, llm_pa_dict, fact_id, text, head, relation, tail,
                        se_name, search_k, llm_type, llm_model_name, llm_temperature, dataset_name, '', '')
                    if len(validation_dict) > 0:
                        last_answer = validation_dict[len(validation_dict) - 1]['answer']
                        last_context = validation_dict[len(validation_dict) - 1]['context']
                        if last_answer == 'LLM_ServiceUnavailableError' or last_context == 'SE_ServiceUnavailableError':
                            index_upper_bound = index
                            break
            else:
                fact_id = index
                head = row['head']
                relation = row['relation']
                tail = row['tail']
                validation_dict, se_qa_dict, llm_pa_dict = validate_single_fact(
                    validation_dict, se_qa_dict, llm_pa_dict, fact_id, 'text_NA', head, relation, tail,
                    se_name, search_k, llm_type, llm_model_name, llm_temperature, dataset_name, '', '')
                if len(validation_dict) > 0:
                    last_answer = validation_dict[len(validation_dict) - 1]['answer']
                    last_context = validation_dict[len(validation_dict) - 1]['context']
                    if last_answer == 'LLM_ServiceUnavailableError' or last_context == 'SE_ServiceUnavailableError':
                        index_upper_bound = index
                        break
    elif dataset_name == 'okele':
        for index in tqdm(range(len(input_data_selected))):
            time.sleep(0.1)
            if last_answer == 'LLM_ServiceUnavailableError' or last_context == 'SE_ServiceUnavailableError':
                break
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
                validation_dict, se_qa_dict, llm_pa_dict = validate_single_fact(
                    validation_dict, se_qa_dict, llm_pa_dict, fact_id, 'text_NA', head_label, relation_label, tail,
                    se_name, search_k, llm_type, llm_model_name, llm_temperature,
                    dataset_name, head_mid, relation_mid)
                if len(validation_dict) > 0:
                    last_answer = validation_dict[len(validation_dict) - 1]['answer']
                    last_context = validation_dict[len(validation_dict) - 1]['context']
                    if last_answer == 'LLM_ServiceUnavailableError' or last_context == 'SE_ServiceUnavailableError':
                        index_upper_bound = index + index_lower_bound
                        break
    else:
        raise RuntimeError('Dataset name Error: ' + dataset_name)

    if index_lower_bound < index_upper_bound:
        # 以excel存储事实验证dict
        validation_excel_path = os.path.join(
            output_directory,
            'fvrel_' + llm_type + '_' + llm_temperature + '_top-' + '{}'.format(search_k) +
            '_from_' + '{}'.format(index_lower_bound) + '_to_' + '{}'.format(index_upper_bound) + '.xlsx'
        )
        da.save_to_excel(validation_dict, validation_excel_path)
        # # 删除'fact_id', 'validation_id', 'text', 'se_call_count', 'llm_call_count'字段，以csv格式存储待标注数据
        # annotation_csv_path = output_directory + '\\4annotation_fvrel_' + llm_type + '_' + llm_model_name + '_' + \
        #                       llm_temperature + '_top-' + '{}'.format(search_k) + \
        #                       '_from_' + '{}'.format(index_lower_bound) + \
        #                       '_to_' + '{}'.format(index_upper_bound) + '.csv'
        # remove_keys = ['fact_id', 'validation_id', 'text', 'se_call_count', 'llm_call_count']
        # for key in remove_keys:
        #     for index in range(len(validation_dict)):
        #         del validation_dict[index][key]
        # save_to_csv(validation_dict, annotation_csv_path)
        # save search engine qa log to reduce the number of api calls
        with open(se_qa_file_path, 'w', encoding='utf-8') as f:
            json.dump(se_qa_dict, f)
            print('File generated successfully...' + se_qa_file_path)
        # save llm prompt answer log to reduce the number of api calls
        with open(llm_pa_file_path, 'w', encoding='utf-8') as f:
            json.dump(llm_pa_dict, f)
            print('File generated successfully...' + llm_pa_file_path)

    return 0


def validate_facts_recursively(dataset_name: str, index_lower_bound: int, index_upper_bound: int,
                               se_name: str, search_k: int, llm_type: str, llm_model_name: str, llm_temperature: str):
    input_file_path, output_directory = get_input_file_and_output_directory(dataset_name)

    method = 'fvrel'
    cur_upper = get_current_upper_bound(output_directory, method, llm_type, llm_temperature, search_k, '', '')
    while cur_upper < index_upper_bound:
        if cur_upper > index_lower_bound:
            validate_facts(dataset_name, cur_upper, index_upper_bound, se_name, search_k,
                           llm_type, llm_model_name, llm_temperature)
        else:
            validate_facts(dataset_name, index_lower_bound, index_upper_bound, se_name, search_k,
                           llm_type, llm_model_name, llm_temperature)
        cur_upper = get_current_upper_bound(output_directory, method, llm_type, llm_temperature, search_k, '', '')
        time.sleep(1)

    return 0


if __name__ == '__main__':
    index_lower_bound = 0
    # 3098 for robot, 3000 for duie, 3060 for okele
    index_upper_bound = 3060
    # fixed to 'google', and use google-serper api
    se_name = 'google'
    # [gpt3.5, gpt4, baichuan2, chatglm-chat6b]

    dataset_name_list = ['okele']
    for dataset_name in dataset_name_list:
        llm_type_list = ['baichuan2-server', 'gpt4']
        for llm_type in llm_type_list:
            llm_model_name = get_llm_model_name(llm_type)

            search_k_list = [1, 5]
            for search_k in search_k_list:
                llm_temp_list = ['0']
                for llm_temperature in llm_temp_list:
                    print(
                        'Start processing ' + dataset_name + '_fvrel' + '_' + llm_type + '_' + llm_temperature +
                        '_top-' + str(search_k) + '_from_' + str(index_lower_bound) + '_to_' + str(index_upper_bound)
                    )
                    validate_facts_recursively(dataset_name, index_lower_bound, index_upper_bound,
                                               se_name, search_k, llm_type, llm_model_name, llm_temperature)
