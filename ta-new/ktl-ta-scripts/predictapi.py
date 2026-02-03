from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
from collections import OrderedDict
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Titles (for the 7 evaluation types)
titles = [
    '喜びを感じた',
    '恐怖を感じた',
    '驚きを感じた',
    '信頼できる情報と感じた',
    '曖昧な情報と感じた',
    '何かの意図をもって書かれたと感じた',
    '経済に期待がもてると感じた',
]

# Initialize FastAPI
app = FastAPI(title="Analyze API", description="Emotion and Sentiment Analysis API", version="1.0.0")


# Input and output data models
class AnalyzeRequest(BaseModel):
    texts: List[str]  # A list of texts to analyze


class AnalyzeResponse(BaseModel):
    predictions: List[Dict[str, float]]  # A list of dictionaries with predictions for each title


# Load models and tokenizers for all 7 titles
def load_all_models(model_dir: str, device: torch.device):
    models = {}
    tokenizers = {}
    for i in range(1, 8):
        try:
            curr_model_dir = os.path.join(model_dir, str(i))
            config = AutoConfig.from_pretrained(curr_model_dir)
            tokenizer = AutoTokenizer.from_pretrained(curr_model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(curr_model_dir, config=config).to(device)
            models[i] = model
            tokenizers[i] = tokenizer
        except Exception as e:
            print(f"Failed to load model {i} from {curr_model_dir}: {e}")
    return models, tokenizers


# Load the models and tokenizers globally to save memory
model_dir = "tamodels/"  # Replace with your actual model directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models, tokenizers = load_all_models(model_dir, device)


# Preprocessing helper function
def preprocess_texts_with_juman(juman, texts):
    try:
        processed = []
        for text in texts:
            cleaned = text.replace("^", "＾")
            res = juman.analysis(cleaned)
            tokens = [mrph.midasi for mrph in res.mrph_list()]
            processed.append(" ".join(tokens))
        return processed
    except Exception as e:
        print(f"Error in Juman preprocessing: {e}")
        return texts  # Fallback: use raw text
#*******************************
def batch_predict(sentences: List[str], model, tokenizer, device) -> List[float]:
    tokenized_inputs = tokenizer(sentences, padding=True, return_tensors="pt")
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

    # Make predictions
    with __import__('torch').no_grad():
        outputs = model(**tokenized_inputs)

    # Convert logits to NumPy array
    predictions = outputs.logits.detach().cpu().numpy()

    # Squeeze extra dimensions
    if len(predictions.shape) == 1:
        predicitons=[predictions]
    else:
        predictions = np.squeeze(predictions,axis=-1) #sqeeseze last dimension safely

    # Ensure the result is always a list, even for single inputs
    if np.isscalar(predictions):
        return [float(predictions)]
    return [float(pred) for pred in predictions]
'''
# Batch prediction function
def batch_predict(texts, model, tokenizer, device):
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    predictions = outputs.logits.detach().cpu().numpy()
    return np.squeeze(predictions).tolist()

'''
# API endpoint for text analysis
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze text input using the loaded models and return predictions.
    """
    # Check if models are loaded
    if not models or not tokenizers:
        raise HTTPException(status_code=500, detail="Models are not properly loaded")

    # Input texts
    texts = request.texts
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided for analysis")

    # Predictions
    responses = []
    for text in texts:
        text_predictions = {}
        for i , title in enumerate(titles):
            model = models[i+1]
            tokenizer = tokenizers[i+1]
            predictions = batch_predict([text], model, tokenizer, device)
            text_predictions[title] = predictions[0]
            #if not isinstance(text_predictions,list):
            #   raise ValueError("bach predicition must be a list")
            #text_predictions[titles[i - 1]] = predictions[0]
        responses.append(text_predictions)
    return {"predictions": responses}
    #return AnalyzeResponse(predictions=responses)
#    except Exception as ex:
#        print (f"[ERROR]  {ex})
#        raise HTTPException(status_code=500, detail=str(ex))
