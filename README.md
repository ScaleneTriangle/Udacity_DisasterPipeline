# Disaster Response Pipeline Project

### Introduction
This project is for the Udacity Data Scientist Nanodegree. When run following the instructions below, 
the output will be a data dashboard at http://0.0.0.0:3001/ that displays some dataset features and will try to
determine the key features of a text in relation to different reponsibilities of disaster response organizations.

My contributions: 
	- The ETL pipeline for cleaning the data.
	- The ML pipeline for building and training the classifier.
	- The Categories bar graph.
	- Extremely small changes to the go.html file to interface with my contributions.
All other pieces of this project were provided by Udacity.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Presentation improvements to explore:
1. Modify the master.html file to not cut the labels of my graph of the bottom.
2. Added descriptions on the limitations of the dataset due to minimal labeled examples for certain categories (Child Alone, Tools, etc)


### Effectiveness Improvements to explore:
1. Try different classifiers (at least SVC).
2. Create additional features for training .
3. Run GridSearchCV on a larger feature space.
