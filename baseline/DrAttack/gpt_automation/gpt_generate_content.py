import os
import ast
import json
import argparse
import time
import pandas as pd
from templates import templates
import requests
from tqdm import tqdm
from utils.test_utils import text_process

def prompts_csv_to_list(path):

    csv_file = pd.read_csv(path)
    prompts = {}
    for goal in csv_file["goal"]:
        prompts[goal] = 1
    return prompts

class DrAttack_prompt_semantic_parser():
    def __init__(self, parsing_tree_dict) -> None:

        self.words_type = []            # a list to store phrase type
        self.words = []                 # a list to store phrase
        self.words_level = []           # a list to store phrase level
        self.words_substitution = []
        
        self.parsing_tree = parsing_tree_dict

    def process_parsing_tree(self):

        self.words_categorization(self.parsing_tree)
        self.words_to_phrases()
        
        for idx, word in enumerate(self.words):
            if self.words_type[idx] == "verb" or self.words_type[idx] == "noun":
                self.words_substitution.append(word) 

    def words_categorization(self, dictionary, depth=0):
        depth += 1
        for key, value in dictionary.items():

            if isinstance(value, str):
                if ("Verb" in key and "Modal" not in key) or ("Gerund" in key) or ("Infinitive" in key):
                    # process Verb labels
                    if depth == 2:
                        # main Verb keeps in how question
                        self.words_type.append("instruction")
                    else:
                        self.words_type.append("verb")
                elif "Determiner" in key:
                    # process Determiner labels
                    if depth == 3:
                        self.words_type.append("instruction")
                    else:
                        self.words_type.append("structure")
                elif "Adjective" in key:
                    # process Adjective labels
                    if depth == 3:
                        self.words_type.append("instruction")
                    else:
                        self.words_type.append("noun")
                elif "Noun" in key:
                    # process Noun labels
                    if depth == 3:
                        self.words_type.append("instruction")
                    elif value == "how":
                        self.words_type.append("structure")
                    else:
                        self.words_type.append("noun")
                elif "Modal Verb" in key:
                    self.words_type.append("structure")
                elif "Relative Pronoun" or "Conj" in key:
                    self.words_type.append("structure")
                elif "how to" or "Infinitive" or 'to' in key:
                    self.words_type.append("structure")
                elif "Preposition" in key:
                    self.words_type.append("structure")
                elif "Adverb" in key:
                    self.words_type.append("structure")
                self.words.append(value)
                self.words_level.append(depth)
            
            if isinstance(value, dict):
                self.words_categorization(value, depth)

    def words_to_phrases(self):
        assert len(self.words_type) == len(self.words)

        idx = 0
        while idx < len(self.words_type) - 1:
            
            if self.words_type[idx] == 'structure' and self.words_type[idx + 1] == 'noun' and self.words_level[idx] == self.words_level[idx+1]:
                self.words[idx] = self.words[idx] + " " + self.words[idx+1]
                self.words_type[idx] = self.words_type[idx + 1]
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
            elif self.words_type[idx] == "instruction" and self.words_type[idx + 1] == "instruction":
                self.words[idx] = self.words[idx] + " " + self.words[idx+1]
                self.words_type[idx] = self.words_type[idx + 1]
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
            elif self.words_type[idx] == "structure" and self.words_type[idx + 1] == "structure" and self.words_level[idx] == self.words_level[idx+1]:
                self.words[idx] = self.words[idx] + " " + self.words[idx+1]
                self.words_type[idx] = self.words_type[idx + 1]
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
            elif self.words_type[idx] == "noun" and self.words_type[idx + 1] == "noun" and self.words_level[idx] == self.words_level[idx+1]:
                self.words[idx] = self.words[idx] + " " + self.words[idx+1]
                self.words_type[idx] = self.words_type[idx + 1]
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
            elif self.words_type[idx] == "verb" and self.words_type[idx + 1] == "verb" and self.words_level[idx] == self.words_level[idx+1]:
                self.words[idx] = self.words[idx] + " " + self.words[idx+1]
                self.words_type[idx] = self.words_type[idx + 1]
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
            else:
                idx += 1
        idx = 0
        while idx < len(self.words_type) - 1:
            
            if self.words_type[idx] == 'noun' and self.words_type[idx + 1] == 'noun':
                self.words[idx] = self.words[idx] + " " + self.words[idx+1]
                self.words_type[idx] = self.words_type[idx + 1]
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
            if self.words_type[idx] == 'structure' and self.words_type[idx + 1] == 'structure':
                self.words[idx] = self.words[idx] + " " + self.words[idx+1]
                self.words_type[idx] = self.words_type[idx + 1]
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
            if self.words_type[idx] == 'verb' and self.words_type[idx + 1] == 'verb':
                self.words[idx] = self.words[idx] + " " + self.words[idx+1]
                self.words_type[idx] = self.words_type[idx + 1]
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
            else:
                idx += 1

class GPT_automation():

    def __init__(self, model="gpt-3.5-turbo", save_path=None):
        self.model = model
        self.save_path = save_path
        self.data = self.load_data(save_path)
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "YOUR_KEY_HERE"
        }

    def load_data(self, file_path):
        if file_path:
            with open(file_path, 'r') as file:
                return json.load(file)
        return {}



    def get_chatgpt_response(self, prompt, verbose=True, presence_penalty=0, frequency_penalty=0,
                             num_retries=20, wait=5):
        # Same implementation as before
        if verbose:
            print(f'Calling ChatGPT. Input length: {len(prompt)}')
        url = self.url
        headers = self.headers
        conv = [{"role": "system", "content": "You are a helpful assistant." },
                {"role": "user", "content": prompt}]
        cdata = {
            "model": self.model,
            "messages": conv,
            "request_timeout": 50
        }
        while True:
            try:
                # ret = openai.ChatCompletion.create(
                #     model=self.model,
                #     messages=[{"role": "system", "content": "You are a helpful assistant." },
                #         {"role": "user", "content": prompt}],
                #     presence_penalty=presence_penalty,
                #     frequency_penalty=frequency_penalty,
                #     request_timeout=50
                # )
                
                response = requests.post(url, headers=headers, data=json.dumps(cdata))
                response_json = response.json()
                print("\n\nresponse_json",response_json)
                ret = response_json
                print("output", ret["choices"][0]["message"]["content"])
                break
            except Exception as e:
                if num_retries == 0:
                    raise RuntimeError
                num_retries -= 1
                print(f'[ERROR] {e}.\nWait for {wait} seconds and retry...')
                time.sleep(wait)
                wait = 50

        return ret["choices"][0]["message"]["content"]

    def save_data(self):
        with open(self.save_path, 'w') as file:
            json.dump(self.data, file, indent=4)

    def process_harmless(self, prompt, prompt_substituable_phrases, templates, generate_mode):

        substitutable_parts = "'" + "', '".join(prompt_substituable_phrases) + "'"

        input_prompt = templates[generate_mode].replace("{user request}", prompt)
        input_prompt = input_prompt.replace("{substitutable parts}", substitutable_parts)
        trial = 0
        valid = False
        while trial <= 10 and not valid:
            try:    
                response = self.get_chatgpt_response(input_prompt)
                word_mapping = ast.literal_eval(response)
                valid = True
            except Exception as e:
                trial += 1

        self.data[prompt][generate_mode] = word_mapping

    def process_opposite(self, prompt, prompt_substituable_phrases, templates, generate_mode):

        self.data[prompt][generate_mode] = {}

        for sub_word in prompt_substituable_phrases:

            input_prompt = templates[generate_mode] + sub_word
            response = self.get_chatgpt_response(input_prompt)
            self.data[prompt][generate_mode][sub_word] = response.split(", ")

    def process_synonym(self, prompt, prompt_substituable_phrases, templates, generate_mode):

        self.data[prompt][generate_mode] = {}

        for sub_word in prompt_substituable_phrases:

            input_prompt = templates[generate_mode] + sub_word
            response = self.get_chatgpt_response(input_prompt)
            self.data[prompt][generate_mode][sub_word] = response.split(", ")

    def process_decomposition(self, prompt, prompt_id, templates, generate_mode):
        trial = 0
        get_response = False
        parsing_tree_dictonary = {}
        while not get_response and trial <= 10:

            input_prompt = templates[generate_mode] + '\"' + prompt + '\"'
            response = self.get_chatgpt_response(input_prompt)
            seg = response.replace("'", "")
            seg = response.replace("```json", "")
            seg = response.replace("```", "")
            trial += 1
            try:
                parsing_tree_dictonary = json.loads(seg)
                get_response = True
            except:
                get_response = False

        self.data[prompt] = {
            'parsing_tree_dictionary': parsing_tree_dictonary,
            'prompt_id': prompt_id,
            'prompt': prompt
        }


    def automate(self, prompts, templates, generate_mode, offset=0, total_number=520):

        prompt_id = 0
        for prompt in prompts:
            prompt_id += 1
            if prompt_id <= offset + total_number and prompt_id >= offset:

                if generate_mode == "harmless":

                    self.process_harmless(prompt, self.data[prompt]["substitutable"], templates, generate_mode)

                elif generate_mode == "opposite":

                    self.process_opposite(prompt, self.data[prompt]["substitutable"], templates, generate_mode)

                elif generate_mode == "synonym":

                    self.process_synonym(prompt, self.data[prompt]["substitutable"], templates, generate_mode)

                elif generate_mode == "decomposition":

                    self.process_decomposition(prompt, prompt_id, templates, generate_mode)

                elif generate_mode == "joint":
                    if prompt not in self.data:
                        self.process_decomposition(prompt, prompt_id, templates, "decomposition")
                    # if not self.data[prompt]['substitutable']:
                    if 'substitutable' not in self.data[prompt]:
                        # self.data[prompt] = {}
                        parser = DrAttack_prompt_semantic_parser(self.data[prompt]["parsing_tree_dictionary"])
                        parser.process_parsing_tree()

                        self.data[prompt]["substitutable"] = parser.words_substitution
                        self.data[prompt]["words"] = parser.words
                        self.data[prompt]["words_level"] = parser.words_level
                        self.data[prompt]["words_type"] = parser.words_type

                        self.process_synonym(prompt, self.data[prompt]["substitutable"], templates, "synonym")
                        self.process_opposite(prompt, self.data[prompt]["substitutable"], templates, "opposite")

                        self.process_harmless(prompt, self.data[prompt]["substitutable"], templates, "harmless")

                        print(f'Saving prompt {prompt_id}')

                        self.save_data()
                else:
                    raise ValueError("Input generation mode not implemented!")
                

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--prompt_path", default="./data/MaliciousInstruct.csv", type=str)
    parser.add_argument("--model", default="gpt-4", type=str)
    parser.add_argument("--generate_mode", default="joint", type=str)
    parser.add_argument("--save_path", default="./attack_prompt_data/MaliciousInstruct_info.json", type=str)
    parser.add_argument("--offset", default=0, type=int)
    parser.add_argument("--total_number", default=520, type=int)
    args = parser.parse_args()

    prompts = prompts_csv_to_list(args.prompt_path)
    args.total_number = len(prompts)
    automation = GPT_automation(model=args.model, save_path=args.save_path)
    automation.automate(prompts, templates, args.generate_mode, offset=args.offset, total_number=args.total_number)