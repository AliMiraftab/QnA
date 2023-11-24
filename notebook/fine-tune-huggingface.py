''' Fine-tune Distil BERT for Exteractive Q&A (train a model to predict start and endpoint in the context to exteract as the answer).
HuggingFace Tools:
1. Datasets: format: Data + Metrics for Evaluation. Type: datasets.arrow_dataset.Dataset from Apache Arrow Table > hashing table for data address
2. Tokenizer: Preprocessing
3. Transformers: Pre-trained model checkpoints +  Trainer Object

Dataset: TyDi QA > https://ai.google.com/research/tydiqa
Model: "distilbert-base-cased-distilled-squad"
'''

from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer

MODEL = "distilbert-base-cased-distilled-squad"

def load_data():
  train_data = load_dataset('tydiqa', 'primary_task')
  tydiqa_data = train_data.filter(lambda example: example['language'] == 'english')
  return tydiqa_data

def load_data_from_disk(path):
  '''
  # Download the dataset from the bucket.
  !wget https://storage.googleapis.com/nlprefresh-public/tydiqa_data.zip
  # Uncomment if you want to check the size of the file. It should be around 319M.
  !ls -alh tydiqa_data.zip
  # Unzip inside the dataset folder
  !unzip tydiqa_data
  '''
  return load_from_disk(path)

def dic2table(data, num_sample=False):
  if num_samples:
    return data.flatten().select(range(num_samples))
  return data.flatten()

# TODO: Preprocessing and tokenizer
def tokenize():
  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  return 
  







