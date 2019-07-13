# stockify
A small app used to display predictions of future stock market changes.

#### Regarding the machine learning module:

In ML, there is a jupyter notebook used to train the model, which provides code snippets and insight towards the development of the REST
API and the Keras model, saved as a HDF5 file.

The server is a Flask application with one route, localhost:5000/predict, that expects a JSON like the response you get when calling 
the alphavantage.co API, namely just the object containing the dates.
The response is a json containing a tag named "values" with an array of prices and a date tag with dates.

To deploy it on localhost, you need to download Anaconda Navigator, import the environment described in the environment.yml file into
conda, activate it in Anaconda Prompt with ''activate stockify'' , then navigate to the ''Server'' folder and launch it into execution with
''python server.py''


