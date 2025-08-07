import requests

from config import get_language_model_api_key, get_language_model_url


class LanguageModelHandler:
    def __init__(self):
        self.api_key = get_language_model_api_key()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.url = get_language_model_url()
    
    def ask_language_model(self, input: str):
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }

        response = requests.post(self.url, headers=self.headers, json=data)
        
        if (not response or 
            response.status_code is not 200):
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
        
        reply = response.json()
        return reply['choices'][0]['message']['content']
