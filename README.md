# Disaster-Response
<img width="808" alt="disaster" src="https://user-images.githubusercontent.com/18635146/108637785-d1df8280-7494-11eb-83bc-f89694abe0e1.PNG">

## Table of Contents
1. [Motivation](#Motivation)
2. [Instructions](#Instructions)
	1. [Files structure](#files_structure)
	2. [Execution](#execution)
	4. [Dependecies](#Dependecies)
3. [Authors](#author)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Screenshots](#Screenshots)

<a name="Motivation"></a>
## Motivation
This project is a part of the Udacity's Data Scientist Nanodegree. 
Using data from Figure-8, the goal is to classify a message that was created in a disaster among 36 categories to help the aid efforts

<a name="files_structure"></a>
## Files structure
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

<a name="Instructions"></a>
## Instructions:

<a name="execution"></a>
### Run the scripts:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database: 
        * `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves it: 
        * `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app: `python run.py`
    - Go to http://0.0.0.0:3001/

<a name="Dependecies"></a>
## Dependecies
- Python 3.8+
- Machine Learning: NumPy, Pandas, Sciki-Learn
- Natural Language Process (NLP): NLTK
- SQLlite: SQLalchemy
- Web App: Flask
- Data Visualization: Plotly

<a name="author"></a>
## Authors

* [Alexandros Evangelou](https://github.com/Evaggelou)

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Figure-Eight](https://www.figure-eight.com/) for providing the dataset
* [Udacity](https://www.udacity.com/)

<a name="Screenshots"></a>
## Screenshots
An example of message "Need some water and food"
<img width="715" alt="message" src="https://user-images.githubusercontent.com/18635146/108638981-b2e3ef00-749a-11eb-84f4-2fc453913778.PNG">
