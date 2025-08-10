import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load your API key from the .env file
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("--- Available Embedding Models ---")

# Loop through and print the name of each model
# The key is to check for the 'embedContent' method
for m in genai.list_models():
  if 'embedContent' in m.supported_generation_methods:
    print(m.name)

print("--------------------------------")