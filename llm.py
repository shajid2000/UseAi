import os
from langchain_google_genai import GoogleGenerativeAI
from decouple import config



def get_gemini_instance():
    os.environ['GOOGLE_API_KEY'] =config('GOOGLE_API_KEY')
    gemini_model_name = config('GEMINI_MODEL_NAME',default="gemini-2.0-flash")
    llm = GoogleGenerativeAI(model=gemini_model_name)
    return llm
