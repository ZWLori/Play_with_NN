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

def plots(train_costs, val_costs, test_costs, l, epochs=1000):
    #Plots
    plt.figure()
    plt.plot(range(epochs), train_costs, label='train costs')
    plt.plot(range(epochs), val_costs, label='val costs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Train and Val Errors at l = %.5f'%l)
    ymin = val_costs.min()
    xmin = np.argmin(val_costs)
    text= "val_lost_min={:.3f}, epoch={:}".format(ymin, xmin)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
    plt.gca().annotate(text, xy=(xmin,ymin), xytext=(0.94,0.6), **kw)
    plt.legend()
    plt.savefig(str(l) + '_train_val_mse.png')
    plt.show()

    plt.figure()
    plt.plot(range(epochs), test_costs)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Test Errors at l = %.5f'%l)
    ymin = test_costs.min()
    xmin = np.argmin(test_costs)
    text= "test_cost_min={:.3f}, epoch={:}".format(ymin, xmin)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
    plt.gca().annotate(text, xy=(xmin,ymin), xytext=(0.94,0.6), **kw)
    plt.savefig(str(l) + '_test_error.png')
    plt.show()

def new_model(no_hidden, no_features):
    print(no_hidden)
    w_o.set_value(np.random.randn(no_hidden)*.01)
    b_o.set_value(np.random.randn()*.01)
    w_h1.set_value(np.random.randn(no_features, no_hidden)*.01)
    b_h1.set_value(np.random.randn(no_hidden)*.01)

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

    trainX = scale(trainX, trainX_min, trainX_max)
    testX = scale(testX, trainX_min, trainX_max)
    return trainX, trainY, testX, testY

# main
np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden = 30 #num of neurons in hidden layer 1
learning_rates = [1e-3, 0.5e-3, 1e-4, 0.5e-4, 1e-5]

floatX = theano.config.floatX

trainX, trainY, testX, testY = read_data()

no_features = trainX.shape[1]
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')

# initialize weights and biases for hidden layer(s) and output layer
w_o = theano.shared(np.random.randn(no_hidden)*.01, floatX )
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(no_features, no_hidden)*.01, floatX )
b_h1 = theano.shared(np.random.randn(no_hidden)*0.01, floatX)

# learning rate
alpha = theano.shared(0.001, floatX)

#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
y = T.dot(h1_out, w_o) + b_o

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

#define gradients
dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h]],
        allow_input_downcast=True
        )

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
)

train_cost = np.zeros(epochs)
test_cost = np.zeros(epochs)
test_accuracy = np.zeros(epochs)
val_cost = np.zeros(epochs)
val_accuracy = np.zeros(epochs)

kf = KFold(n_splits=5)

for l in learning_rates:
    alpha.set_value(l)
    print(alpha.get_value())
    test_costs = []
    train_costs = []
    val_costs = []

    for train_index, val_index in kf.split(trainX):
        val_set_X = trainX[val_index]
        val_set_Y = trainY[val_index]
        train_set_X = trainX[train_index]
        train_set_Y = trainY[train_index]
        new_model(no_hidden, no_features)

        for iter in range(epochs):
            if iter % 100 == 0:
                print(iter)

            train_set_X, train_set_Y = shuffle_data(train_set_X, train_set_Y)
            val_set_X, val_set_Y = shuffle_data(val_set_X, val_set_Y)

            train_cost[iter] = train(train_set_X, np.transpose(train_set_Y))
            val_pred, val_cost[iter], val_accuracy[iter] = test(val_set_X, np.transpose(val_set_Y))
            pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

        test_costs.append(test_cost)
        train_costs.append(train_cost)
        val_costs.append(val_cost)

        best_cost = np.min(test_costs)
        best_accuracy = np.max(test_accuracy)

        print('Minimum error: %.1f, Best accuracy %.1f'%(best_cost, best_accuracy))

    plots(np.mean(train_costs, axis=0), np.mean(val_costs, axis=0), np.mean(test_costs, axis=0), l)
