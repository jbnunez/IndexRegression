#berkeleynn.py
import numpy as np
import math, time
import itertools
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation
from keras.layers import Input, Flatten
from keras.models import load_model, Model
import keras
from numpy import genfromtxt
#import seaborn as sns



#learning rate = 0.001
learn_rate = 0.001

#hidden layers
layer_sizes = (100, 150, 250, 100, 50)

#epoch limit
max_epochs = 30

#activation fuction = unipolar sigmoid
act_func = 'sigmoid'

#previous data points
prev_points_considered = 4

leverage_used = 3

#stopping criteria: max epochs, error value increases past a certain amount, 
#or change in error is below a certain threshold
error_threshold = 0.000001
error_max_increase = 0.001

#data points: open, high, low, close, frac_change for past 4 trading days
#22 data points per day
#used sensex index
def preprocess_data():
    data = np.genfromtxt("Nifty data.csv", delimiter=',')

    #Remove the header names
    names, data = data[0], data[1:]

    #Deletes the columns containing Date, time, frac change, and nan value
    data = np.delete(data, [0,1, 6, 7], 1)
    actualPrice = data[:,3]

    #Compute average and standard deviation of the values
    minData = np.min(data, axis=0)
    maxData = np.max(data, axis=0)

    #Normalize all values with min 0, max 1
    vF = lambda x: (x-minData)/(maxData-minData)
    data = vF(data)

    return data, actualPrice[5:]


def get_past_n_points(past_points, data):
    n = past_points
    while n <= len(data): 
        yield data[n - past_points : n]
        n+=1


def get_targets(past_points, data):
    return np.delete(data[past_points-1:], [0,1,2,4],1)


def getBatchSizeNPast(num_points, data):
    new_data = []
    x = get_past_n_points(num_points, data)
    for i in x: 
        new_data.append(i)
    return np.array(new_data)


def build_model(size_list):
    #input is a 4x5 matrix 
    input_layer = Input(shape=(4,4))
    hidden = []
    for i in range(len(size_list)):
        if i == 0:
            prev_layer = input_layer
        else:
            prev_layer = hidden[i-1]

        hidden.append(Dense(size_list[i], activation=act_func, use_bias=True, 
        kernel_initializer='lecun_normal', bias_initializer='zeros', 
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
        kernel_constraint=None, bias_constraint=None)(prev_layer))

    flatten_layer = Flatten()(hidden[len(hidden)-1])

    #output layer simply takes a weighted sum of the outputs 
    #of the last hidden layer
    output_layer = Dense(1, activation=act_func, use_bias=True, 
        kernel_initializer='lecun_normal', bias_initializer='zeros', 
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
        kernel_constraint=None, bias_constraint=None)(flatten_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model


def evaluate_performance(model, x, y):
    y_pred = model.predict(x, batch_size=None, verbose=0, steps=None)
    y_bar = np.mean(y)
    y_pred_bar = np.mean(y_pred)
    sstot = 0
    for yi in y:
        sstot += (yi-y_bar)**2
    ssreg = 0
    for fi in y_pred:
        ssreg += (fi-y_bar)**2
    ssres = 0
    for i in range(len(y)):
        ssres += (y[i]-y_pred[i])**2
    rsquared = 1 - (ssres/sstot)
    print("R^2 value is: ", rsquared)



def simulate_trading(x, y, model, leverage, prices):
    print("We will simulate 50 days of trading")
    print("Each morning, we retrain the network based off of the past 50 days")
    print("Then we predict the price every 15 minutes, and trade off that")
    print("if the price increases by >0.5%, go long in etf")
    print("if the price decreases by >0.5%, go short in etf")
    print("assume $10,000 initial capital")
    backup = 0
    trading_days = 1000
    starting_data = 50*22 #22 data points per day
    capital = 10000.0
    market = 10000.0
    #training length
    tl = 80
    #gains = []
    #truth = []
    for i in range(tl, tl+trading_days):
        capital_open = capital

        if i%50==0:
            x_train = x[(i-tl)*22:i*22]
            y_train = y[(i-tl)*22:i*22]

            model.fit(x=x_train, y=y_train, batch_size=20, epochs=max_epochs, verbose=1, 
            callbacks=None, validation_split=0.2, validation_data=None, 
            shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, 
            steps_per_epoch=None, validation_steps=None)

        x_day = x[i*22:(i+1)*22]
        y_day = y[i*22:(i+1)*22]
        price_day = prices[i*22:(i+1)*22]

        day_open = prices[(i*22) - 1]
        day_close = price_day[-1]
        print("making predictions")
        y_pred = model.predict(x_day, batch_size=None, verbose=0, steps=None)
        #make a decision every 15 minutes
        print("starting trades")
        for j in range(1,len(y_pred)):
            #gains.append(capital[0]/10000)
            #truth.append(y_day[j][0]/y[1100][0])
            print("making a trade:")
            if (y_pred[j]/y_day[j-1]) > 1.005:
                #go long
                print("long")
                capital -= 7
                capital *= ((( (prices[j]/prices[j-1]) - 1)*leverage)+1)
                capital -= 7
            elif (y_pred[j]/y_day[j-1]) < 0.995:
                #go short
                print("short")
                capital -= 7
                capital *= ((( (prices[j-1]/prices[j]) - 1)*leverage)+1)
                capital -= 7
            else:
                print("no trade")
        market = market*(day_close/day_open)
        print("Close trading day ", i-600)
        print("Today portfolio finished at ", (100.0*capital/capital_open) - 100, " percent change from opening value")
        print("capital value: ", capital, ", market: ", market)
        print("backup contains: ", backup)
        if capital > 100000:
            backup += capital - 10000
            capital = 10000
        if capital < 1000:
            capital += 10000
            backup -= 10000
    print("After 50 days, portfolio ended at $", capital)
    #sns.set(style = "ticks")

    return capital/10000

# def plot_data(data1, data2):
#     sns.set(style="ticks")
#     ax = sns.tsplot(data = data1, color = "b", condition= "Predictions")
#     bx = sns.tsplot(data = data2, color = "r", condition = "Ground Truth")
#     plt.show()


def main():
    data, actualPrices = preprocess_data()
    x_all = getBatchSizeNPast(prev_points_considered, data)
    y_all = get_targets(prev_points_considered, data)

    x_test = x_all[600*22:]
    y_test = y_all[600*22:]

    x_train = x_all[:600*22]
    y_train = y_all[:600*22]
    print(len(x_train))

    model = build_model(layer_sizes)

    model.compile(loss='mean_squared_error', optimizer='Adam')
    
    # model.fit(x=x_train, y=y_train, batch_size=20, epochs=max_epochs, verbose=1, 
    #     callbacks=None, validation_split=0.2, validation_data=None, 
    #     shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, 
    #     steps_per_epoch=None, validation_steps=None)
    
    #evaluate_performance(model, x_test, y_test)

    simulate_trading(x_all, y_all, model, leverage_used, actualPrices)
    


main()
