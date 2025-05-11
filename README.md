# P18-Shimmy-Moderation
**University of Manchester - Applying Data Science Project 18**  
Content Moderation Model for the Shimmy App

---

## Project Overview
This project focuses on building a content moderation system for the Shimmy App. It leverages machine learning models to classify and moderate user-generated content based on severity, category, and flagged status.

---

## Requirements
- **Python Version**: `3.12.4`
- **Dependencies**: Listed in `requirements.txt`

---

## Large Language Model (LLM) Details
- **Source**: [Ollama](https://ollama.com/)
- **Model**: [Gemma3:12b](https://ollama.com/library/gemma3:12b)
- **Download Ollama**: [Download Here](https://ollama.com/download)

### Running the LLM
1. Download and install Ollama from the source.
2. Start the model:
   ```bash
   ollama run gemma3:12b
   ```
3. If the model hasn't started, try:
   ```bash
   ollama serve
   ```
   Ensure port 11434 is available for listening (usually free).

---

## Classifier Execution
1. Install the requirements:
   ```bash
   pip install requirements.txt
   ```
2. Download and unzip the models folder from the link provided in `/model/files.md`.
3. Run the main.py file:
   ```bash
   python main.py
   ```

---

## Shimmy Test App
Shimmy Test App is designed by our group to simulate the process of handling comments and illustrating the moderation dashboard.

### Setup Instructions
1. Install the libraries listed in `requirements.txt`.
2. Download the following into your Python environment:
   - `model`
   - `tokenizer`
   - `Shimmy Test App Modification`
   - `data/harm_categories.json`
3. Run the app using the command below:
   ```bash
   streamlit run app_version0423-2.py
   ```
