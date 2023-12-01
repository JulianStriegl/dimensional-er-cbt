import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import wordcloud
import nltk
import unicodedata
import string


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
nltk.download('omw-1.4')

tf.get_logger().setLevel('ERROR')


# Load Data

# Load VAD libabry into lexicon
lexicon_path = "content/Emotion-Detection-Datasets/NRC-VAD-Lexicon.txt"
lex = {}

with open(lexicon_path) as f:
  for line in f:
    words = line.split()
    try:
      lex[words[0]] = [float(words[1]),float(words[2]),float(words[3])]
    except:
      continue

# EmoBank
train_emo_ds = pd.read_csv("content/Emotion-Detection-Datasets/emobank.csv",names=["id", "split", "V", "A", "D","text"],skiprows=1)

def normalize(data):
  return (data-1)/4

# split dataset into v,a,d
emo_data = np.array(train_emo_ds.pop('text'))
emo_v = normalize(np.array(train_emo_ds.pop('V')))
emo_a = normalize(np.array(train_emo_ds.pop('A')))
emo_d = normalize(np.array(train_emo_ds.pop('D')))
emo_vad = np.array([emo_v,emo_a,emo_d]).T

#GoEmotions
raw_data = pd.read_csv("content/Emotion-Detection-Datasets/goemotions.csv")
data_text = raw_data[["text"]]
data_scores = raw_data[["admiration","amusement","anger",
                          "annoyance","approval","caring","confusion",
                          "curiosity","desire","disappointment","disapproval",
                          "disgust","embarrassment","excitement","fear","gratitude",
                          "grief","joy","love","nervousness","optimism","pride",
                          "realization","relief","remorse","sadness","surprise","neutral"]]

train_text = []
labels = ["admiration","amusement","anger",
                          "annoyance","approval","caring","confusion",
                          "curiosity","desire","disappointment","disapproval",
                          "disgust","embarrassment","excitement","fear","gratitude",
                          "grief","joy","love","nervousness","optimism","pride",
                          "realization","relief","remorse","sadness","surprise","neutral"]
text = {}
for entry in data_text.get("text"):
  text[entry] = [0]*28
  train_text.append(entry)
for i in range(len(train_text)):
  train_text[i] = [train_text[i],[0,0,0],0]

for i in range(len(train_text)):
  emotions = []
  for entry in data_scores:
    emotions.append(data_scores.get(entry)[i])
  text[data_text.get("text")[i]] = np.add(text[data_text.get("text")[i]],emotions)

scores = {}
for item in text:
  scores[item] = [[0,0,0],0]
  for x in range(28):
    for i in range(text[item][x]):
      scores[item] = [np.add(scores[item][0],[float(j) for j in lex[labels[x]]]),scores[item][1]+1]

for item in scores:
  try:
    scores[item][0] = [float(x)/scores[item][1] for x in scores[item][0]]
  except:
    del item

# prepare Goemotion data to format
go_data = []
go_v = []
go_a = []
go_d = []
go_vad = []

for sentence in text.keys():
  go_data.append(sentence)
  go_v.append(scores[sentence][0][0])
  go_a.append(scores[sentence][0][1])
  go_d.append(scores[sentence][0][2])
  go_vad.append([scores[sentence][0][0],scores[sentence][0][1],scores[sentence][0][2]])


go_data = np.array(go_data)
go_v = np.array(go_v)
go_a = np.array(go_a)
go_d = np.array(go_d)
go_vad = np.array(go_vad)


#ISEAR Dataset
raw_data = pd.read_csv('content/Emotion-Detection-Datasets/isear.csv', delimiter = '|', on_bad_lines='skip', encoding='ISO-8859-1')
data_text = raw_data[["SIT"]]
data_scores = raw_data[["EMOT"]]

train_text = []

for entry in data_text.get("SIT"):
    train_text.append(entry)
for i in range(len(train_text)):
  train_text[i] = [train_text[i],[0,0,0]]

for i in range(len(train_text)):

  if data_scores.get("EMOT")[i] == 1:
   data_scores.get("EMOT")[i] = "joy"
   train_text[i][1] = [float(i) for i in lex["joy"]]
  if data_scores.get("EMOT")[i] == 2:
   data_scores.get("EMOT")[i] = "fear"
   train_text[i][1] = [float(i) for i in lex["fear"]]
  if data_scores.get("EMOT")[i] == 3:
   data_scores.get("EMOT")[i] = "anger" 
   train_text[i][1] = [float(i) for i in lex["anger"]]
  if data_scores.get("EMOT")[i] == 4:
   data_scores.get("EMOT")[i] = "sadness" 
   train_text[i][1] = [float(i) for i in lex["sadness"]]
  if data_scores.get("EMOT")[i] == 5:
   data_scores.get("EMOT")[i] = "disgust" 
   train_text[i][1] = [float(i) for i in lex["disgust"]]
  if data_scores.get("EMOT")[i] == 6:
   data_scores.get("EMOT")[i] = "shame" 
   train_text[i][1] = [float(i) for i in lex["shame"]]
  if data_scores.get("EMOT")[i] == 7:
   data_scores.get("EMOT")[i] = "guilt" 
   train_text[i][1] = [float(i) for i in lex["guilt"]]

# prepare Goemotion data to format
isear_data = []
isear_v = []
isear_a = []
isear_d = []
isear_vad = []

for i in range(len(train_text)):
  isear_data.append(train_text[i][0])
  isear_v.append(train_text[i][1][0])
  isear_a.append(train_text[i][1][1])
  isear_d.append(train_text[i][1][2])
  isear_vad.append([train_text[i][1][0],train_text[i][1][1],train_text[i][1][2]])

isear_data = np.array(isear_data)
isear_v = np.array(isear_v)
isear_a = np.array(isear_a)
isear_d = np.array(isear_d)
isear_vad = np.array(isear_vad)

#CrowdFlower
raw_data = pd.read_csv('content/Emotion-Detection-Datasets/crowdflower.csv',on_bad_lines='skip')
data_text = raw_data[["content"]]
data_scores = raw_data[["sentiment"]]

train_text = []

for entry in data_text.get("content"):
    train_text.append(entry)
for i in range(len(train_text)):
  train_text[i] = [train_text[i],[0,0,0]]

for i in range(len(train_text)):

   train_text[i][1] = [float(i) for i in lex[ data_scores.get("sentiment")[i]]]

# prepare Goemotion data to format
cf_data = []
cf_v = []
cf_a = []
cf_d = []
cf_vad = []

for i in range(len(train_text)):
  cf_data.append(train_text[i][0])
  cf_v.append(train_text[i][1][0])
  cf_a.append(train_text[i][1][1])
  cf_d.append(train_text[i][1][2])
  cf_vad.append([train_text[i][1][0],train_text[i][1][1],train_text[i][1][2]])

cf_data = np.array(cf_data)
cf_v = np.array(cf_v)
cf_a = np.array(cf_a)
cf_d = np.array(cf_d)
cf_vad = np.array(cf_vad)

# Combine data
combined_data = []
combined_vad = []

combined_data.extend(go_data)
combined_data.extend(emo_data)
combined_data.extend(isear_data)
combined_vad.extend(go_vad)
combined_vad.extend(emo_vad)
combined_vad.extend(isear_vad)

print("Number of samples:",len(combined_vad))

# Load combined_data from the file
# combined_data = np.load('combined_data.npy')

# Load combined_vad from the file
# combined_vad = np.load('combined_vad.npy')

x_train, x_test, vad_train, vad_test = train_test_split(combined_data, combined_vad, test_size=0.02, shuffle= False)
x_train, x_valid, vad_train, vad_valid = train_test_split(x_train, vad_train, test_size=0.2, shuffle= True)


#define optimizer
loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
metrics = tf.metrics.mean_squared_error

epochs = 5
# steps_per_epoch = tf.data.experimental.cardinality(x_train).numpy()
print(len(x_train))
steps_per_epoch = len(x_train)
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

#load model
classifier_model_vad = tf.keras.models.load_model("content/Emotion-Detection-Datasets/all_albertx20.h5",custom_objects={'KerasLayer':hub.KerasLayer})
classifier_model_vad.compile(optimizer=optimizer,loss=loss,metrics=metrics)


#Rule-based Approach

# transform POS to wordnet scheme
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return "a"
    elif treebank_tag.startswith('V'):
        return "v"
    elif treebank_tag.startswith('N'):
        return "n"
    elif treebank_tag.startswith('R'):
        return "r"
    else:
        return 'n'

# negation function for vad scores
def negate(score):

  for i in range(len(score)):
    difference = abs(score[i]-.5)
    if score[i] < .5:
      score[i] += difference*2
    else:
      score[i] -= difference*2
  return score

# Remove accents function
def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters or x == " ")

# Rule based ED
def emotion_detection(input):

  # transform string into list
  input = input.split()

  stopwords = nltk.corpus.stopwords.words('english')
  stemmer = nltk.stem.PorterStemmer()
  lemmatizer = nltk.stem.WordNetLemmatizer()

  # remove accents and punctuation 
  input = [remove_accents(x) for x in input]

  # transform to lowercase
  input = [x.lower() for x in input]

  pos = nltk.pos_tag(input)

  for i in range(len(input)):
    input[i] = lemmatizer.lemmatize(input[i],get_wordnet_pos(pos[i][1]))

  score = [0,0,0]
  total = 0

  for i in range(len(input)):


    if input[i] in lex:
      
      if input[i-1] == ("not" or "never"):
        score = np.add(score,negate([float(i) for i in lex[input[i]]]))
      else:
        score = np.add(score,[float(i) for i in lex[input[i]]])
      total += 1

  if total > 0:
    score = [float(x)/total for x in score]

  return score

# Helper Functions for Evaluation
def mean_squared_error_combined(real,target):
  error=0
  for i in range(len(real)):
    error += abs(real[i]-target[i])
  return (error/len(real))**2

def mean_squared_error_individual(real,target):
  error = abs(real-target)
  return error**2

def map_to_categories_vad(vad_score,categories):
  vad_categories = {}
  differences = {}
  for category in categories:
    vad_categories[category] = lex[category]
  for category in vad_categories.keys():
    differences[category] = np.absolute(np.subtract(vad_categories[category],vad_score))
    mean = 0
    for x in differences[category]:
      mean += x
    mean = mean/len(differences[category])
    differences[category] = mean
  return (min(differences, key=differences.get))

def map_to_categories_va(vad_score,categories):
  vad_categories = {}
  differences = {}
  for category in categories:
    vad_categories[category] = lex[category][:-1]
  for category in vad_categories.keys():
    differences[category] = np.absolute(np.subtract(vad_categories[category],vad_score[:-1]))
    mean = 0
    for x in differences[category]:
      mean += x
    mean = mean/len(differences[category])
    differences[category] = mean
  return (min(differences, key=differences.get))


def evaluate(data,vad):

  # import time
  # start = time.time()

  mse_v = 0
  mse_a = 0
  mse_d = 0
  mse_combined = 0
  ml_predicted = []
  for i in range(len(data)):
    predicted = classifier_model_vad(tf.constant([data[i]])).numpy()
    mse_v += mean_squared_error_individual(predicted[0][0],vad[i][0])
    mse_a += mean_squared_error_individual(predicted[0][1],vad[i][1])
    mse_d += mean_squared_error_individual(predicted[0][2],vad[i][2])
    mse_combined += mean_squared_error_combined(predicted[0],vad[i])
    ml_predicted.append(predicted[0])
  mse_combined_v = mse_v/len(data)
  mse_combined_a = mse_a/len(data)
  mse_combined_d = mse_d/len(data)
  mse_combined_total = mse_combined/len(data)
  coefficient_v = np.corrcoef(np.array(ml_predicted).T[0],np.array(vad).T[0])[0][1]
  coefficient_a = np.corrcoef(np.array(ml_predicted).T[1],np.array(vad).T[1])[0][1]
  coefficient_d = np.corrcoef(np.array(ml_predicted).T[2],np.array(vad).T[2])[0][1]
  coefficient_vad = (coefficient_v+coefficient_a+coefficient_d)/3


  print("V MSE ML-approach:",mse_combined_v)
  print("A MSE ML-approach:",mse_combined_a)
  print("D MSE ML-approach:",mse_combined_d)
  print("Combined MSE ML-approach:",mse_combined_total)
  print("V correlation ml",coefficient_v)
  print("A correlation ml",coefficient_a)
  print("D correlation ml",coefficient_d)
  print("Combined correlation ML-approach:",coefficient_vad)  

  mse_v = 0
  mse_a = 0
  mse_d = 0
  mse_combined = 0
  r_predicted = []
  for i in range(len(data)):
    predicted = emotion_detection(data[i])
    mse_v += mean_squared_error_individual(predicted[0],vad[i][0])
    mse_a += mean_squared_error_individual(predicted[1],vad[i][1])
    mse_d += mean_squared_error_individual(predicted[2],vad[i][2])
    mse_combined += mean_squared_error_combined(predicted,vad[i])
    r_predicted.append(predicted)
  mse_combined_v = mse_v/len(data)
  mse_combined_a = mse_a/len(data)
  mse_combined_d = mse_d/len(data)
  mse_combined_total = mse_combined/len(data)

  # end = time.time()
  # print(end-start)

  coefficient_v = np.corrcoef(np.array(r_predicted).T[0],np.array(vad).T[0])[0][1]
  coefficient_a = np.corrcoef(np.array(r_predicted).T[1],np.array(vad).T[1])[0][1]
  coefficient_d = np.corrcoef(np.array(r_predicted).T[2],np.array(vad).T[2])[0][1]
  coefficient_vad = (coefficient_v+coefficient_a+coefficient_d)/3

  print("V MSE Rule-approach:",mse_combined_v)
  print("A MSE Rule-approach:",mse_combined_a)
  print("D MSE Rule-approach:",mse_combined_d)
  print("Combined MSE Rule-approach:",mse_combined_total)
  print("V correlation rule",coefficient_v)
  print("A correlation rule",coefficient_a)
  print("D correlation rule",coefficient_d)
  print("Combined correlation Rule-approach:",coefficient_vad)
  print("\n")

def visualize_evaluation(data,vad):

  ml_predicted = []
  r_predicted = []
  for i in range(len(data)):
    predicted_ml = classifier_model_vad(tf.constant([data[i]])).numpy()
    predicted_r = emotion_detection(data[i])
    ml_predicted.append(predicted_ml[0])
    r_predicted.append(predicted_r)

  v = [i[0] for i in vad]
  a = [i[1] for i in vad]
  d = [i[2] for i in vad]

  print("Real Data distribution")

  plt.xlabel('Valence')
  plt.ylabel('Arousal')
  plt.scatter(v, a)
  plt.savefig("VA-Real",transparent = True,bbox_inches='tight',dpi=300)
  plt.show()

  plt.xlabel('Valence')
  plt.ylabel('Dominance')
  plt.scatter(v, d)
  plt.savefig("VD-Real",transparent = True,bbox_inches='tight',dpi=300)
  plt.show()

  plt.xlabel('Arousal')
  plt.ylabel('Dominance')
  plt.scatter(a, d)
  plt.savefig("AD-Real",transparent = True,bbox_inches='tight',dpi=300)
  plt.show()

  v = [word[0] for word in ml_predicted]
  a = [word[1] for word in ml_predicted]
  d = [word[2] for word in ml_predicted]

  print("ML-inferred Data distribution")

  plt.xlabel('Valence')
  plt.ylabel('Arousal')
  plt.scatter(v, a)
  plt.savefig("VA-ML",transparent = True,bbox_inches='tight',dpi=300)
  plt.show()
  plt.xlabel('Valence')
  plt.ylabel('Dominance')
  plt.scatter(v, d)
  plt.savefig("VD-ML",transparent = True,bbox_inches='tight',dpi=300)
  plt.show()

  plt.xlabel('Arousal')
  plt.ylabel('Dominance')
  plt.scatter(a, d)
  plt.savefig("AD-ML",transparent = True,bbox_inches='tight',dpi=300)
  plt.show()

  v = [word[0] for word in r_predicted]
  a = [word[1] for word in r_predicted]
  d = [word[2] for word in r_predicted]

  print("Rule-inferred Data distribution")

  plt.xlabel('Valence')
  plt.ylabel('Arousal')
  plt.scatter(v, a)
  plt.savefig("VA-Rule",transparent = True,bbox_inches='tight',dpi=300)
  plt.show()
  plt.xlabel('Valence')
  plt.ylabel('Dominance')
  plt.scatter(v, d)
  plt.savefig("VD-Rule",transparent = True,bbox_inches='tight',dpi=300)
  plt.show()
  plt.xlabel('Arousal')
  plt.ylabel('Dominance')
  plt.scatter(a, d)
  plt.savefig("AD-Rule",transparent = True,bbox_inches='tight',dpi=300)
  plt.show()

def evaluate_categorical_ml(data,vad):

  categories = {"empty":[0,0,0,0,0,0,0],"threatened":[0,0,0,0,0,0,0],"tranquil":[0,0,0,0,0,0,0],"excited":[0,0,0,0,0,0,0],"rooted":[0,0,0,0,0,0,0]}
  tp = 0
  total = 0
  dataset_distribution_categorical = {}
  for key in categories.keys():
    dataset_distribution_categorical[key]=0

  metrics = {}
  for category in categories:
    metrics[category] = {"TP":0,"TN":0,"FP":0,"FN":0,"Precision":0,"Recall":0,"F1":0}


  for i in range(len(data)):
    prediction = map_to_categories_va(classifier_model_vad(tf.constant([data[i]])).numpy()[0],categories.keys())
    real = map_to_categories_va(vad[i],categories.keys())
    dataset_distribution_categorical[real] += 1
    if prediction == real:
      tp +=1
      total += 1
      for category in metrics.keys():
        if prediction == category:
          metrics[category]["TP"] += 1
        else:
          metrics[category]["TN"] += 1

          
    else:
      total += 1
      for category in metrics.keys():
        if prediction == category:
          metrics[category]["FP"] += 1
        elif real == category:
          metrics[category]["FN"] += 1
        else:         
          metrics[category]["TN"] += 1

  for category in metrics.keys():
    if metrics[category]["TP"]==0:
      metrics[category]["Precision"] = 0
      metrics[category]["Recall"] = 0
      metrics[category]["F1"] = 0
    else:
      metrics[category]["Precision"] = metrics[category]["TP"]/(metrics[category]["TP"]+metrics[category]["FP"])
      metrics[category]["Recall"] = metrics[category]["TP"]/(metrics[category]["TP"]+metrics[category]["FN"])
      metrics[category]["F1"] = 2*metrics[category]["Precision"]*metrics[category]["Recall"]/(metrics[category]["Precision"]+metrics[category]["Recall"])

  macro = 0
  total_tp = 0
  total_fp = 0
  total_fn = 0
  average = 0

  for category in metrics.keys():
    macro +=   metrics[category]["F1"]
    total_tp += metrics[category]["TP"]
    total_fp += metrics[category]["FP"]
    total_fn += metrics[category]["FN"]

  macro = macro/len(metrics.keys())
  micro = total_tp/(total_tp+.5*(total_fp+total_fn))
  for category in categories:
    average += (dataset_distribution_categorical[category]/total)*metrics[category]["F1"]
  print(metrics)
  print("accuracy =",tp/total)
  print("macro f1 =",macro)
  print("micro f1 =",micro)
  print("average f1 =",average)

def evaluate_categorical_r(data,vad):

  categories = {"empty":[0,0,0,0,0,0,0],"threatened":[0,0,0,0,0,0,0],"tranquil":[0,0,0,0,0,0,0],"excited":[0,0,0,0,0,0,0],"rooted":[0,0,0,0,0,0,0]}
  tp = 0
  total = 0
  dataset_distribution_categorical = {}
  for key in categories.keys():
    dataset_distribution_categorical[key]=0

  metrics = {}
  for category in categories:
    metrics[category] = {"TP":0,"TN":0,"FP":0,"FN":0,"Precision":0,"Recall":0,"F1":0}


  for i in range(len(data)):
    prediction = map_to_categories_va(emotion_detection(data[i]),categories.keys())
    real = map_to_categories_va(vad[i],categories.keys())
    dataset_distribution_categorical[real] += 1
    if prediction == real:
      tp +=1
      total += 1
      for category in metrics.keys():
        if prediction == category:
          metrics[category]["TP"] += 1
        else:
          metrics[category]["TN"] += 1

          
    else:
      total += 1
      for category in metrics.keys():
        if prediction == category:
          metrics[category]["FP"] += 1
        elif real == category:
          metrics[category]["FN"] += 1
        else:         
          metrics[category]["TN"] += 1

  for category in metrics.keys():
    if metrics[category]["TP"]==0:
      metrics[category]["Precision"] = 0
      metrics[category]["Recall"] = 0
      metrics[category]["F1"] = 0
    else:
      metrics[category]["Precision"] = metrics[category]["TP"]/(metrics[category]["TP"]+metrics[category]["FP"])
      metrics[category]["Recall"] = metrics[category]["TP"]/(metrics[category]["TP"]+metrics[category]["FN"])
      metrics[category]["F1"] = 2*metrics[category]["Precision"]*metrics[category]["Recall"]/(metrics[category]["Precision"]+metrics[category]["Recall"])

  macro = 0
  total_tp = 0
  total_fp = 0
  total_fn = 0
  average = 0

  for category in metrics.keys():
    macro +=   metrics[category]["F1"]
    total_tp += metrics[category]["TP"]
    total_fp += metrics[category]["FP"]
    total_fn += metrics[category]["FN"]

  macro = macro/len(metrics.keys())
  micro = total_tp/(total_tp+.5*(total_fp+total_fn))
  for category in categories:
    average += (dataset_distribution_categorical[category]/total)*metrics[category]["F1"]
  print(metrics)
  print("accuracy =",tp/total)
  print("macro f1 =",macro)
  print("micro f1 =",micro)
  print("average f1 =",average)

print("Evaluation against Emobank:")
evaluate(emo_data,emo_vad)

print("Evaluation against GoEmotion:")
evaluate(go_data,go_vad)

print("Evaluation against ISEAR:")
evaluate(isear_data,isear_vad)

print("Evaluation against Combined:")
evaluate(combined_data,combined_vad)

print("Evaluation against Split:")
evaluate(x_test,vad_test)

print("Evaluation against CrowdFlower:")
evaluate(cf_data,cf_vad)

visualize_evaluation(emo_data,emo_vad)


# vizualize categorical evaluation
# scheme: emotion: [tp,tn,fp,fn,precision,recall,f1]
categories = {"empty":[0,0,0,0,0,0,0],"threatened":[0,0,0,0,0,0,0],"tranquil":[0,0,0,0,0,0,0],"excited":[0,0,0,0,0,0,0],"rooted":[0,0,0,0,0,0,0]}
v = []
a = []
colors = ["blue","red","green","orange","pink"]

for entry in categories:
  v.append(lex[entry][0])
  a.append(lex[entry][1])

fig, ax = plt.subplots()
plt.xlabel('Valence')
plt.ylabel('Arousal')

ax.annotate("empty", (v[0], a[0]),xytext=(0.22, 0.16))
ax.annotate("threatened", (v[1], a[1]),xytext=(0.1, 0.9))
ax.annotate("tranquil", (v[2], a[2]),xytext=(0.77, 0.08))
ax.annotate("excited", (v[3], a[3]),xytext=(0.77, 0.9))
ax.annotate("rooted", (v[4], a[4]),xytext=(0.55, 0.5))


for i in range(len(colors)):
  plt.scatter(v[i],a[i],color = colors[i])

plt.savefig("Categorization",transparent = True,bbox_inches='tight',dpi=300)
plt.show()

colors = {"empty":"blue","threatened":"red","tranquil":"green","excited":"orange","rooted":"pink"}
v = []
a = []
c = []


fig, ax = plt.subplots()

  
plt.xlabel('Valence')
plt.ylabel('Arousal')

score = []
units = 30
for i in range(units):
  for j in range(units):
    score.append([[i/units,j/units,1],colors[map_to_categories_va([i/units,j/units,1],categories.keys())]])


for i in range(len(score)):
  # print(score[i][0][0],score[i][0][1],score[i][1])
  plt.scatter(score[i][0][0],score[i][0][1],color = score[i][1])

plt.savefig("Categorization_distance",transparent = True,bbox_inches='tight',dpi=300)

dataset_distribution_categorical = {}
total = 0

for key in categories.keys():
  dataset_distribution_categorical[key]=0

for datapoint in combined_vad:
  dataset_distribution_categorical[map_to_categories_va(datapoint,categories.keys())] += 1
  total += 1

for key in categories.keys():
  dataset_distribution_categorical[key]=dataset_distribution_categorical[key]/total

print("Categorical Distribution of the train dataset:", dataset_distribution_categorical)


# Categorical Evaluation
print("Combined:")
evaluate_categorical_ml(combined_data,combined_vad)
evaluate_categorical_r(combined_data,combined_vad)
print("CrowdFlower:")
evaluate_categorical_ml(cf_data,cf_vad)
evaluate_categorical_r(cf_data,cf_vad)
print("GoEmotion:")
evaluate_categorical_ml(go_data,go_vad)
evaluate_categorical_r(go_data,go_vad)
print("ISEAR:")
evaluate_categorical_ml(isear_data,isear_vad)
evaluate_categorical_r(isear_data,isear_vad)

#loss, mean_squared_error = classifier_model_vad.evaluate(x=x_valid,y=vad_valid)

#print("VAD: ")
#print(f'Loss: {loss}')
#print(f'mean squared error: {mean_squared_error}')