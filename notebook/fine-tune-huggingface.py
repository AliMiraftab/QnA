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
  return tokenizer

# Processing samples using the 3 steps described.
def process_samples(sample):
    tokenized_data = tokenizer(sample['document_plaintext'], sample['question_text'], truncation="only_first", padding="max_length")

    input_ids = tokenized_data["input_ids"]

    # We will label impossible answers with the index of the CLS token.
    cls_index = input_ids.index(tokenizer.cls_token_id)

    # If no answers are given, set the cls_index as answer.
    if sample["annotations.minimal_answers_start_byte"][0] == -1:
        start_position = cls_index
        end_position = cls_index
    else:
        # Start/end character index of the answer in the text.
        gold_text = sample["document_plaintext"][sample['annotations.minimal_answers_start_byte'][0]:sample['annotations.minimal_answers_end_byte'][0]]
        start_char = sample["annotations.minimal_answers_start_byte"][0]
        end_char = sample['annotations.minimal_answers_end_byte'][0] #start_char + len(gold_text)

        # sometimes answers are off by a character or two – fix this
        if sample['document_plaintext'][start_char-1:end_char-1] == gold_text:
            start_char = start_char - 1
            end_char = end_char - 1     # When the gold label is off by one character
        elif sample['document_plaintext'][start_char-2:end_char-2] == gold_text:
            start_char = start_char - 2
            end_char = end_char - 2     # When the gold label is off by two characters

        start_token = tokenized_data.char_to_token(start_char)
        end_token = tokenized_data.char_to_token(end_char - 1)

        # if start position is None, the answer passage has been truncated
        if start_token is None:
            start_token = tokenizer.model_max_length
        if end_token is None:
            end_token = tokenizer.model_max_length

        start_position = start_token
        end_position = end_token

    return {'input_ids': tokenized_data['input_ids'],
          'attention_mask': tokenized_data['attention_mask'],
          'start_positions': start_position,
          'end_positions': end_position}

def transformers():
  from transformers import AutoModelForQuestionAnswering
  model = AutoModelForQuestionAnswering.from_pretrained(MODEL)
  return model

def compute_f1_metrics(pred):
  from sklearn.metrics import f1_score
  
  start_labels = pred.label_ids[0]
  start_preds = pred.predictions[0].argmax(-1)
  end_labels = pred.label_ids[1]
  end_preds = pred.predictions[1].argmax(-1)

  f1_start = f1_score(start_labels, start_preds, average='macro')
  f1_end = f1_score(end_labels, end_preds, average='macro')

  return {
    'f1_start': f1_start,
    'f1_end': f1_end,
  }  

def training(model):
  from transformers import Trainer, TrainingArgumnets

  training_args = TrainingArguments(
    output_dir='model_results5',          # output directory
    overwrite_output_dir=True,
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=20,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=None,            # directory for storing logs
    logging_steps=50)

  trainer = Trainer(
    model=model, # the instantiated 🤗 Transformers model to be trained
    args=training_args, # training arguments, defined above
    train_dataset=processed_train_data, # training dataset
    eval_dataset=processed_test_data, # evaluation dataset
    compute_metrics=compute_f1_metrics)

  trainer.train()    
  print(trainer.evaluate(processed_test_data))
  return model

def test_samples(model):
  import torch

  text = r"""
  The Golden Age of Comic Books describes an era of American comic books from the
  late 1930s to circa 1950. During this time, modern comic books were first published
  and rapidly increased in popularity. The superhero archetype was created and many
  well-known characters were introduced, including Superman, Batman, Captain Marvel
  (later known as SHAZAM!), Captain America, and Wonder Woman.
  Between 1939 and 1941 Detective Comics and its sister company, All-American Publications,
  introduced popular superheroes such as Batman and Robin, Wonder Woman, the Flash,
  Green Lantern, Doctor Fate, the Atom, Hawkman, Green Arrow and Aquaman.[7] Timely Comics,
  the 1940s predecessor of Marvel Comics, had million-selling titles featuring the Human Torch,
  the Sub-Mariner, and Captain America.[8]
  As comic books grew in popularity, publishers began launching titles that expanded
  into a variety of genres. Dell Comics' non-superhero characters (particularly the
  licensed Walt Disney animated-character comics) outsold the superhero comics of the day.[12]
  The publisher featured licensed movie and literary characters such as Mickey Mouse, Donald Duck,
  Roy Rogers and Tarzan.[13] It was during this era that noted Donald Duck writer-artist
  Carl Barks rose to prominence.[14] Additionally, MLJ's introduction of Archie Andrews
  in Pep Comics #22 (December 1941) gave rise to teen humor comics,[15] with the Archie
  Andrews character remaining in print well into the 21st century.[16]
  At the same time in Canada, American comic books were prohibited importation under
  the War Exchange Conservation Act[17] which restricted the importation of non-essential
  goods. As a result, a domestic publishing industry flourished during the duration
  of the war which were collectively informally called the Canadian Whites.
  The educational comic book Dagwood Splits the Atom used characters from the comic
  strip Blondie.[18] According to historian Michael A. Amundson, appealing comic-book
  characters helped ease young readers' fear of nuclear war and neutralize anxiety
  about the questions posed by atomic power.[19] It was during this period that long-running
  humor comics debuted, including EC's Mad and Carl Barks' Uncle Scrooge in Dell's Four
  Color Comics (both in 1952).[20][21]
  """
  
  questions = ["What superheroes were introduced between 1939 and 1941 by Detective Comics and its sister company?",
               "What comic book characters were created between 1939 and 1941?",
               "What well-known characters were created between 1939 and 1941?",
               "What well-known superheroes were introduced between 1939 and 1941 by Detective Comics?"]
  
  for question in questions:
      inputs = tokenizer.encode_plus(question, text, return_tensors="pt")
      print("inputs", inputs)
      print("inputs", type(inputs))
      input_ids = inputs["input_ids"].tolist()[0]
      inputs.to("cuda")
  
      text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
      answer_model = model(**inputs)
  
      answer_start = torch.argmax(
          answer_model['start_logits']
      )  # Get the most likely beginning of answer with the argmax of the score
      answer_end = torch.argmax(answer_model['end_logits']) + 1  # Get the most likely end of answer with the argmax of the score
  
      answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
  
      print(f"Question: {question}")
      print(f"Answer: {answer}\n")



if __name__ == '__main__':

  tokenizer = tokenize()
  # Tokenizing and processing the flattened dataset
  processed_train_data = flattened_train_data.map(process_samples)
  processed_test_data = flattened_test_data.map(process_samples)

  # Transformers
  columns_to_return = ['input_ids','attention_mask', 'start_positions', 'end_positions']
  processed_train_data.set_format(type='pt', columns=columns_to_return)
  processed_test_data.set_format(type='pt', columns=columns_to_return)

  model = training(transformers())
  test_samples(model)
  
  
  





