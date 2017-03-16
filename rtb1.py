#import graphlab module
import graphlab as gl

#read training data CSV to variable 'data'
data = gl.SFrame.read_csv('train.csv', verbose=False)

#read validation data CSV to variable 'testdata'
testdata = gl.SFrame.read_csv('validation.csv', verbose=False)

#create logisic regression classifier from chosen features
#data = train.csv
#click is target
model = gl.logistic_classifier.create(data, target='click', features=['advertiser', 'useragent', 'slotheight', 'slotwidth'], max_iterations=20, validation_set=None)

#model2 = gl.recommender.factorization_recommender.create(data, target='click')
#model2 = gl.factorization_recommender.create(data, target='click', user_id='userid', item_id='slotid')
model2 = gl.boosted_trees_regression.create(data, target='click', features=['advertiser', 'useragent', 'slotheight', 'slotwidth'], max_iterations=20, validation_set=None)
model3 = gl.random_forest_regression.create(data, target='click', features=['advertiser', 'useragent', 'slotheight', 'slotwidth'], max_iterations=20, validation_set=None)
model4 = gl.decision_tree_regression.create(data, target='click', features=['advertiser', 'useragent', 'slotheight', 'slotwidth'], validation_set=None)
#net = gl.deeplearning.get_builtin_neuralnet('mnist')
#model4 = gl.neuralnet_classifier.create(data, target='click', network=net, max_iterations=20, validation_set=None)

#predict the PCTR values against the validation dataset,
#testdata = validation.csv
#values = predicted values

values = model.predict(testdata, output_type='probability')
values2 = model2.predict(testdata)
values3 = model3.predict(testdata)
values4 = model4.predict(testdata)
#values4 = model4.predict(testdata)

#test model prediction, prints first 5 rows
print values.head(5)
print values2.head(5)
print values3.head(5)
print values4.head(5)
#print values4.head(5)

#add values as a column
#values = values from the prediction
#newdata is the new variable for this new Sframe
#testdata is our validation dataset
newdata = testdata.add_column(values, name='PCTR_logistic')
newdata2 = testdata.add_column(values2, name='PCTR_GBRT')
newdata3 = testdata.add_column(values3, name = "PCTR_RFG ")
newdata4 = testdata.add_column(values4, name = "PCTR_Linear ")
#newdata4 = testdata.add_column(values4, name="PCTR_Neural")

#SAVING
newdata.save('validationprediction.csv', format='csv')
newdata2.save('validationprediction.csv', format='csv')
newdata3.save('validationprediction.csv', format='csv')
newdata4.save('validationprediction.csv', format='csv')
#newdata4.save('validationprediction.csv', format='csv')