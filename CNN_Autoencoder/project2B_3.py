from load import mnist
import numpy as np
import pylab
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)

def annotate_min(pylab, y):
    ymin = min(y)
    xmin = np.argmin(y)
    text= "min={:.3f}, epoch={:}".format(ymin, xmin)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
    pylab.gca().annotate(text, xy=(xmin,ymin), xytext=(0.94,0.6), **kw)

def plot_learning_curves(layer_no, train_da):
    print('training layer' + str(layer_no) + ' of denoising autoencoder ...')
    reconstruction_err = []
    for epoch in range(training_epochs):
        # go through trainng set
        cost = []
        # mini-batch
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            cost.append(train_da(trX[start:end]))
        reconstruction_err.append(np.mean(cost, dtype='float64'))

    # plot the learning curves for layer
    pylab.figure()
    pylab.plot(range(training_epochs), reconstruction_err)
    annotate_min(pylab, reconstruction_err)
    pylab.xlabel('iterations')
    pylab.ylabel('cross-entropy')
    pylab.savefig('learning_curve_'+str(layer_no)+'.png')
    pylab.show()

def plot_weights(layer_no, W, size):
    w = W.get_value()
    pylab.figure()
    pylab.gray()
    for i in range(100):
        pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w[:,i].reshape(size,size))
    pylab.savefig('w' + str(layer_no) + '.png')

def plot_activations(activation, layer_no, size):
    pylab.figure()
    pylab.gray()
    for i in range(100):
        pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(activation[i,:].reshape(size,size))
    pylab.savefig('activation' + str(layer_no) + '.png')
    pylab.show()

def sgd_momentum(cost, params, lr=0.1, momentum=0.1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value()*0.)
        v_new = momentum*v - g * lr 
        updates.append([p, p + v_new])
        updates.append([v, v_new])
    return updates

trX, teX, trY, teY = mnist()

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]

x = T.fmatrix('x')
d = T.fmatrix('d')

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

corruption_level=0.1
training_epochs = 25
learning_rate = 0.1
batch_size = 128
beta = 0.5
rho = 0.05
gamma = 0.1

# 3 hidden layers
W1 = init_weights(28*28, 900)
b1 = init_bias(900)
b1_prime = init_bias(28*28)
W1_prime = W1.transpose()

W2 = init_weights(900, 625)
b2 = init_bias(625)
b2_prime = init_bias(900)
W2_prime = W2.transpose()

W3 = init_weights(625, 400)
b3 = init_bias(400)
b3_prime = init_bias(625)
W3_prime = W3.transpose()

# output layer
W4 = init_weights(400, 10)
b4 = init_bias(10)

# train first hidden layer
# corrupting inputs -- binomial distribution
tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x
y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
# cross-entropy & sparsity constraint
cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))            + beta*T.shape(y1)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho))             - beta*rho*T.sum(T.log(T.mean(y1, axis=0)+1e-6))             - beta*(1-rho)*T.sum(T.log(1-T.mean(y1, axis=0)+1e-6))

params1 = [W1, b1, b1_prime]
grads1 = T.grad(cost1, params1)
train_da1 = theano.function(inputs=[x], outputs = cost1, updates = sgd_momentum(cost1, params1, learning_rate, gamma), allow_input_downcast = True)

plot_learning_curves(1, train_da1)
plot_weights(1, W1, 28)

# train second hidden layer
# corrupting inputs -- binomial distribution
tilde_y1 = theano_rng.binomial(size=y1.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*y1
y2 = T.nnet.sigmoid(T.dot(tilde_y1, W2) + b2)
z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)
# cross-entropy
cost2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1))            + beta*T.shape(y2)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho))             - beta*rho*T.sum(T.log(T.mean(y2, axis=0)+1e-6))             - beta*(1-rho)*T.sum(T.log(1-T.mean(y2, axis=0)+1e-6))

params2 = [W2, b2, b2_prime]
grads2 = T.grad(cost2, params2)
train_da2 = theano.function(inputs=[x], outputs = cost2, updates = sgd_momentum(cost2, params2, learning_rate, gamma), allow_input_downcast = True)

plot_learning_curves(2, train_da2)
plot_weights(2, W2, 30)

# train third hidden layer
# corrupting inputs -- binomial distribution
tilde_y2 = theano_rng.binomial(size=y2.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*y2
y3 = T.nnet.sigmoid(T.dot(tilde_y2, W3) + b3)
z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)
# cross-entropy
cost3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1))            + beta*T.shape(y3)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho))             - beta*rho*T.sum(T.log(T.mean(y3, axis=0)+1e-6))             - beta*(1-rho)*T.sum(T.log(1-T.mean(y3, axis=0)+1e-6))

params3 = [W3, b3, b3_prime]
grads3 = T.grad(cost3, params3)
train_da3 = theano.function(inputs=[x], outputs = cost3, updates = sgd_momentum(cost3, params3, learning_rate, gamma), allow_input_downcast = True)

plot_learning_curves(3, train_da3)
plot_weights(3, W3, 25)

# construct stacked autoencoder
encoder1 = theano.function(inputs=[x], outputs = y1, allow_input_downcast=True)
encoder2 = theano.function(inputs=[y1], outputs = y2, allow_input_downcast=True)
encoder3 = theano.function(inputs=[y2], outputs = y3, allow_input_downcast=True)

decoder3 = theano.function(inputs=[y3], outputs = z3, allow_input_downcast=True)
decoder2 = theano.function(inputs=[y2], outputs = z2, allow_input_downcast=True)
decoder1 = theano.function(inputs=[y1], outputs = z1, allow_input_downcast=True)

# random select 100 imgs from test set
idx = np.random.randint(teX.shape[0], size=100)
test_100 = theano.function(inputs=[x], outputs = [y1,y2,y3], allow_input_downcast=True)
activation1, activation2, activation3 = test_100(teX[idx,:])

# plot the activation for each hidden layer
plot_activations(activation1, 1, 30)
plot_activations(activation2, 2, 25)
plot_activations(activation3, 3, 20)

# original images
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(teX[idx[i],:].reshape(28,28))
pylab.savefig('original.png')
pylab.show()

# reconstructed images
regenerated = decoder1(decoder2(decoder3(encoder3(encoder2(encoder1(teX[idx,:]))))))
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(regenerated[i,:].reshape(28,28))
pylab.savefig('reconstructed.png')
pylab.show()


# question 2 : train a 5-layer feedforward NN using the pretrained weights for the 3 hidden layers

p_y = T.nnet.softmax(T.dot(y3, W4)+b4)
y4 = T.argmax(p_y, axis=1)
cost4 = T.mean(T.nnet.categorical_crossentropy(p_y, d))        + beta*T.shape(p_y)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho))         - beta*rho*T.sum(T.log(T.mean(p_y, axis=0)+1e-6))         - beta*(1-rho)*T.sum(T.log(1-T.mean(p_y, axis=0)+1e-6))

params4 = [W1, b1, W2, b2, W3, b3, W4, b4]
grads4 = T.grad(cost4, params4)
updates4 = [(param4, param4 - learning_rate * grad4)
           for param4, grad4 in zip(params4, grads4)]
train_ffn = theano.function(inputs=[x, d], outputs = cost4, updates = sgd_momentum(cost4, params4, learning_rate, gamma), allow_input_downcast = True)
test_ffn = theano.function(inputs=[x], outputs = y4, allow_input_downcast=True)

print('\ntraining ffn ...')
train_err, test_acc = [], []
for epoch in range(training_epochs):
    # go through trainng set
    cost = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost.append(train_ffn(trX[start:end], trY[start:end]))
    train_err.append(np.mean(cost, dtype='float64'))
    test_acc.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))

fig = pylab.figure()
pylab.plot(range(training_epochs), train_err)
annotate_min(pylab, train_err)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.savefig('train_err.png')
pylab.show()

pylab.figure()
pylab.plot(range(training_epochs), test_acc)
ymax = max(test_acc)
xmax = np.argmax(test_acc)
text= "test_acc_max={:.3f}, epoch={:}".format(ymax, xmax)
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
kw = dict(xycoords='data',textcoords="axes fraction",
          arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
pylab.gca().annotate(text, xy=(xmax,ymax), xytext=(0.94,0.6), **kw)
pylab.xlabel('iterations')
pylab.ylabel('test accuracy')
pylab.savefig('test_acc.png')
pylab.show()
