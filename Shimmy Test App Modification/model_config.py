import ollama
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import json
import os
import re
from pathlib import Path

class SentimentClassifier:

    def __init__(self):
        base_path = Path(__file__).parent  # Get the directory of the current file
        model_path = base_path / "../model"  # Adjust the relative path to the model directory
        tokenizer_path = base_path / "../tokenizer"

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.action_dict = {
            0: 'Needs Moderation',
            1: 'User Approval Needed',
            2: 'Safe / Not Harmful, No action needed'
        }
        self.class_dict = {
            0: 'Negative',
            1: 'Slightly Negative / Neutral',
            2: 'Positive'
        }

    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def get_sentiment_from_roberta(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        # scores = output[0][0].cpu().detach().numpy()
        scores = softmax(scores)
        predicted_category = np.argmax(scores)
        return predicted_category
    
    def get_sentiment(self, text):
        cleaned_text = self.preprocess(text)
        prompt = f"""
        Classify the following text into one of the three categories (0, 1, 2):
        - Category 0: Moderately or Strongly Negative
        - Category 1: Neutral or Slightly Negative
        - Category 2: Safe, Positive, Not harmful

        Text: "{cleaned_text}"

        Respond with only a single integer from 0 to 2.
        """
        
        response = ollama.chat(model="gemma3:12b", messages=[{"role": "user", "content": prompt}])
        try:
            classified_category = int(response["message"]["content"].strip())
        except ValueError:
            classified_category = self.get_sentiment_from_roberta(cleaned_text)
            
        return classified_category 

    def predict(self, text):
        flag = int(self.get_sentiment(text))
        category = self.class_dict[flag]
        action = self.action_dict[flag]

        return flag, category, action
    

class Reasoner:
    def __init__(self, text, file_path):
        self.text = text
        self.offense_types_str = json.dumps(json.load(open(file_path, 'r')), indent=4)

    def generate_response(self):
        prompt = f"""
        Match the given negative statement with one of the given categories and give reason in strictly <50 words:
        - Statement: {self.text}
        - Categories: {self.offense_types_str}
        """
        
        response = ollama.chat(model="gemma3:12b", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip() 


if __name__=='__main__':

    offense_types_file_path = Path(__file__).parent.parent / "data" / "harm_categories.json"
    classifier = SentimentClassifier()
    text = input("Enter Comment: ")
    flag, category, action = classifier.predict(text)
    
    print("Comment: ", text)
    print(f"Category: {category}\nAction: {action}")

    if flag==0:
        reasoner = Reasoner(text, offense_types_file_path)
        response = reasoner.generate_response()
        category_match = re.search(r"\*\*Category:\*\*\s*(.+)", response)
        reason_match = re.search(r"\*\*Reason:\*\*\s*([\s\S]+)", response)
        category = category_match.group(1).strip() if category_match else ""
        reason = reason_match.group(1).strip() if reason_match else ""
        print("Category: ", category)
        print("Reason: ", reason)
        # print("Reason: ", response)






