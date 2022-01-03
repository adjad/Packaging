import tensorflow as tf
import pandas
from flask import Flask

#app=Flask(__main__)


model=tf.keras.models.load_model('the_model')
datatest = pandas.read_csv("Packaging Data - Test Data.csv")
labelstest=datatest.pop('Package Answer')
datatest.pop('Material type')
datatest.pop('Returns')
datatest.pop('Item #')
datatest.pop('No.')
print(datatest)
#model.evaluate(datatest,labelstest, verbose=2)


#def 

#if __name__ == "__main__":
#    app.run()