# dimensional-er-cbt
Dimensional ALBERT-based model for dimensional text-based emotion recognition for conversational agent-based cognitive behavioral therapy.
Fine-tuned model can be found in the content folder ("all_albertx20.h5").
Used datasets can be found in the content folder.

## Evaluation Metrics and Statistics
- User study evaluation results can be found in the folder evaluation -> user_study_evaluation_results.xslt
- Results of the technical evaluation (mean squared error, correlation coefficient, f1 score etc.) can be found in evaluation -> technical_evaluation_results.txt
- Use the eval.py file to compile the fine-tuned model and run the evaluation yourself
- Generated plots can be found in the plots folder

## DL- & Rule-based Emotion Detection (includes training, datasets), Emotion Mapping scheme and technical Evaluation

### Create virtualenv and activate
pip install virtualenv: python -m virtualenv venv

to activate on windows: cd venv/Scripts/
cmd activate.bat -> activate

to deactivate: deactivate

alternatively: .\venv\activate

### Install requirements and run notebook
- install requirements.txt: pip install -r requirements.txt
- run the jupyter notebook "Emotion_Detection_Rule_vs_ML.ipynb" and select venv as runtime environment
- alternatively you can run eval.py to test and evaluate the model in comparison to a rule based approach

This work was created by Jordan Wenzel Richter under the supervision of Julian Striegl as part of the Junior Research Group: "AI-based Coaching of Students/AI-based CBT" of the Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI) Dresden/Leipzig.