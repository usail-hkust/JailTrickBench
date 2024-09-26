import time
import os
import torch
import requests
import json
from utils.test_utils import text_process


class Text_Embedding_Ada():

    def __init__(self, model="text-embedding-ada-002"):

        self.model = model
        self.load_saved_embeddings()

    def load_saved_embeddings(self):
        self.embedding_cache = {}

    def get_embedding(self, text):

        text = text.replace("\n", " ")

        ## memoization
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        ret = None
        retry_num = 0
        while ret is None:
            try:
                # response = openai.Embedding.create(input=[text],
                #             model=self.model,
                #             request_timeout=10)['data'][0]['embedding']
                url = "https://api.openai.com/v1/embeddings" 
                headers = { 
                "Content-Type": "application/json", 
                "Authorization": "Bearer YOUR_KEY_HERE" 
                } 
                new_text = text_process(text)
                data = { 
                "input": new_text,
                "encoding_format":"float" #encoding_format support "float"and "base64"
                } 
                response_json = requests.post(url, headers=headers, data=json.dumps(data)) 
                response_json = response_json.json()
                response = response_json['data'][0]['embedding']
                ret = torch.tensor(response).unsqueeze(0).unsqueeze(0)

            except Exception as e:
                print(e)
                retry_num += 1
                if retry_num > 5:
                    print('Failed to get embedding for text: \n')
                    print("=*"*20)
                    print(text)
                    print("=*"*20)
                    
                if 'rate limit' in str(e).lower():  ## rate limit exceed
                    print('wait for 20s and retry...')
                    time.sleep(20)
                else:
                    print('Retrying... ', retry_num)
                    time.sleep(5)
    
        self.embedding_cache[text] = ret

        return ret