# P18-Shimmy-Moderation
UoM Applying Data Science Project 18 - Content Moderation Model - Shimmy App

```
Python: 3.12.4
```

### LLM Details:

Source: https://ollama.com/

Download: https://ollama.com/download

Model: https://ollama.com/library/gemma3:12b

### Ollama Execution:
Download and Install Ollama from the source

```
ollama run gemma3:12b
```

if the model hasn't started try:

```
ollama serve
```

The port 11434 should be available for listening (usually free)


### Classifier Execution:
Install the requirements

```
pip install requirements.txt
```

Download and unzip the models folder from the link provided in /model/files.md

Run the main.py file

```
python main.py
```

### Shimmy Test App
Shimmy Test App is designed by our group to simulate the process of handling comments. 

Note: You should install the libraries in requirements.txt. Download `model`, `tokenizer`,`Shimmy Test App Modification` into your python environment. Download `data/harm_categories.json`.

Type command below in the terminal:
```
streamlit run app.py
```
