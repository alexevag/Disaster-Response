# Disaster-Response
## Motivation
This project is a part of the Udacity's Data Scientist Nanodegree. 
Using data from Figure-8, the goal is to classify a message that was created in a disaster among 36 categories to help the aid efforts

## The file structure
```
- app  
| - template  
| |- master.html  # main page of web app  
| |- go.html  # classification result page of web app  
|- run.py  # Flask file that runs app  

- data  
|- disaster_categories.csv # data to process   
|- disaster_messages.csv # data to process  
|- process_data.py  
|- DisasterResponse.db # database

- models  
|- train_classifier.py  
|- classifier.pkl  # the saved model

- README.md  
```

## Instructions:

# Run the scripts:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database: 
        * `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves it: 
        * `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app: `python run.py`

## Dependecies
- Python 3.8+
- Machine Learning: NumPy, Pandas, Sciki-Learn
- Natural Language Process (NLP): NLTK
- SQLlite: SQLalchemy
- Web App: Flask
- Data Visualization: Plotly

## Screenshots
