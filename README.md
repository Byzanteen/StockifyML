# stockify
A small app used to display predictions of future stock market changes.

In ML, there is a jupyter notebook used to train the model, which provides code snippets and insight towards the development of the REST
API and the Keras model, saved as a HDF5 file.

The server is a Flask application with one route, localhost:5000/predict, that expects a JSON like the response you get when calling 
the alphavantage.co API, namely just the object containing the dates.
The response is a json containing a tag named "values" and an array of values that are not tagged with the date they represent but are sorted
by the date.They need to be tagged with the date in the back-end (or in the flask app).

