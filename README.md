# dimensional-er-cbt
Dimensional ALBERT-based model for dimensional text-based emotion recognition for conversational agent-based cognitive behavioral therapy.


## User Study Evaluation Statistics

- can be found as Excel table in Evaluation.xlsx

## DL- & Rule-based Emotion Detection (includes training, datasets), Emotion Mapping scheme and technical Evaluation

### create virtualenv and activate
pip install virtualenv
python -m virtualenv venv

to activate on windows: cd venv/Scripts/
cmd activate.bat -> activate
to deactivate: deactivate

alternatively: .\venv\activate

### Install Requirements and Run Notebook
- install requirements.txt: pip install -r requirements.txt
- run the jupyter notebook "Emotion_Detection_Rule_vs_ML.ipynb" and select venv as runtime environment
- alternatively you can run eval.py to test and evaluate the model in comparison to a rule based approach

This work was created by Jordan Wenzel Richter under the supervision of Julian Striegl as part of the Junior Research Group: "AI-based Coaching of Students/AI-based CBT" of the Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI) Dresden/Leipzig.
