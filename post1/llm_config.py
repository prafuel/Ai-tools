import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI

# =======================================================
from langchain_core.runnables import RunnableLambda
# =======================================================   

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from azure.ai.inference.models import SystemMessage, UserMessage

load_dotenv()

# ============================================================================= #

def load_azure_openai(model_name: str = "gpt-4o-ins1"):
    # initialization
    API_KEY = os.environ['AZURE_API_KEY']
    VERSION = os.environ['AZURE_API_VERSION']
    AZURE_API_BASE = os.environ['AZURE_API_BASE']

    azure_chat_openai = AzureChatOpenAI(
        azure_deployment=model_name,
        azure_endpoint=AZURE_API_BASE,
        api_key=API_KEY,
        api_version=VERSION,
        model=f"azure/{model_name}"
    )

    return azure_chat_openai

# ============================================================================= #

def load_gemini(model_name: str = "gemini-2.0-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model_name
    )

    return llm

# ============================================================================= #

def load_groq(model_name: str = "mistral-saba-24b"):
    llm = ChatGroq(
        model=model_name
    )

    return llm

# ============================================================================= #

def load_mistral(model_name: str = "mistral-large2411-fai-doc"):
    endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
    deployment_name = model_name
    key = os.getenv("AZURE_INFERENCE_SDK_KEY")
    client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    def invoke(prompt: str, **kwargs):
        try:
            response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant. just return only what asked"),
                UserMessage(content=prompt)
            ],
            max_tokens=2048,
                temperature=0.8,
                top_p=0.1,
                model = deployment_name,
            )
            return response['choices'][0]['message']
        except Exception as e:
            error_json = json.dumps({
                "complex_word_map" : {"error" : str(e)},
                "updated_sentence" : str(e)
            })
            return {
                "content" : f"""```json \n{error_json}\n```"""
            }
    return RunnableLambda(lambda x: invoke(x['text'] if isinstance(x, dict) else x.text)['content'])

# ============================================================================= #

def load_huggingface(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        tokenizer, model = None, None
    
    def invoke(prompt: str, **kwargs):
        try:
            if tokenizer is None or model is None:
                raise Exception("Model not loaded properly")
            
            # Get parameters from kwargs or use defaults
            max_length = kwargs.get('max_length', 200)
            temperature = kwargs.get('temperature', 0.8)
            top_p = kwargs.get('top_p', 0.1)
            
            inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
            
            outputs = model.generate(
                **inputs, 
                # max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False
            )
            
            text = tokenizer.batch_decode(outputs)[0]
            
            return {
                "content": text
            }
            
        except Exception as e:
            error_json = json.dumps({
                "complex_word_map": {"error": str(e)},
                "updated_sentence": str(e)
            })
            return {
                "content": f"""```json \n{error_json}\n```"""
            }
    
    return RunnableLambda(lambda x: invoke(x['text'] if isinstance(x, dict) else x.text)['content'])
