import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold

# scale input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

def plots(test_costs, epochs=1000):
    #Plots
    plt.figure()
    plt.plot(range(epochs), test_costs)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Test Error for 4-layer NN')
    ymin = test_costs.min()
    xmin = np.argmin(test_costs)
    text= "test_cost_min={:.3f}, epoch={:}".format(ymin, xmin)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
    plt.gca().annotate(text, xy=(xmin,ymin), xytext=(0.94,0.6), **kw)
    plt.savefig('4_layer_test_error.png')
    plt.show()

def read_data():
    #read and divide data into test and train sets
    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
    X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
    Y_data = (np.asmatrix(Y_data)).transpose()

    X_data, Y_data = shuffle_data(X_data, Y_data)

    #separate train and test data
    m = 3*X_data.shape[0] // 10
    testX, testY = X_data[:m],Y_data[:m]
    trainX, trainY = X_data[m:], Y_data[m:]

    # scale data
    trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
    testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

    trainX = scale(trainX, trainX_min, trainX_max)
    testX = scale(testX, trainX_min, trainX_max)
    return trainX, trainY, testX, testY

# main
np.random.seed(10)

epochs = 1000
no_hidden1 = 50 #num of neurons in hidden layer 1
no_hidden2 = 20
learning_rate = 0.0001

floatX = theano.config.floatX

trainX, trainY, testX, testY = read_data()

no_features = trainX.shape[1]
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')

# initialize weights and biases for hidden layer(s) and output layer
w_o = theano.shared(np.random.randn(no_hidden2)*.01, floatX )
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
b_h1 = theano.shared(np.random.randn(no_hidden1)*.01, floatX)
w_h2 = theano.shared(np.random.randn(no_hidden1, no_hidden2)*.01, floatX )
b_h2 = theano.shared(np.random.randn(no_hidden2)*.01, floatX)

# learning rate
alpha = theano.shared(learning_rate, floatX)

#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
y = T.dot(h2_out, w_o) + b_o

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

#define gradients
dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2 = T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2])

train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h1],
                   [b_h1, b_h1 - alpha*db_h1],
                   [w_h2, w_h2 - alpha*dw_h2],
                   [b_h2, b_h2 - alpha*db_h2]],
        allow_input_downcast=True)

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
)

train_cost = np.zeros(epochs)
test_cost = np.zeros(epochs)
test_accuracy = np.zeros(epochs)

print(alpha.get_value())

# train model
for iter in range(epochs):
    if iter % 100 == 0:
        print(iter)

    train_cost[iter] = train(trainX, np.transpose(trainY))
    pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

plots(test_cost)
