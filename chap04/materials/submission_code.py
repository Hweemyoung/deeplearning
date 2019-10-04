
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import copy

x_train, y_train, x_test = load_fashionmnist()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1000)

def relu(x):
    return np.maximum(x, 0)

def deriv_relu(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    x -= x.max(axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def deriv_softmax(x):
    return softmax(x) * (1 - softmax(x))

class Dense:
    def __init__(self, in_dim, out_dim, function, deriv_function):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype('float64')
        self.b = np.zeros([out_dim]).astype('float64')
        
        self.function = function
        self.deriv_function = deriv_function
        
        self.x = None
        self.u = None
        
        self.dW = None
        self.db = None
        
def f_props(layers, x):
    h = x
    for layer in layers:
        layer.x = h
        u = layer.u = np.matmul(layer.x, layer.W) + layer.b
        h = layer.h = layer.function(u)
    
    return h

def b_props(layers, delta, lam=0):
    batch_size = delta.shape[0]
    
    for i, layer in enumerate(layers[::-1]):
        if i > 0: delta = layer.deriv_function(layer.u) * np.matmul(delta, W.T)

        layer.dW = np.matmul(layer.x.T, delta) / batch_size  + lam * layer.W
        layer.db = np.mean(delta, 0)

        W = layer.W

def update_parameters(layers, eps):
    for layer in layers:
        layer.W -= eps*layer.dW
        layer.b -= eps*layer.db

layers = [
    Dense(784, 100, relu, deriv_relu),
    Dense(100, 100, relu, deriv_relu),
    Dense(100, 10, softmax, deriv_softmax)
]

def train_mnist(x, t, eps, lam):
    y = f_props(layers, x)
    delta = (y - t)
    
    b_props(layers, delta, lam)
    update_parameters(layers, eps)
    
def valid_mnist(x, t, layers):
    y = f_props(layers, x)
    y_pred = np.argmax(y, axis=1)
    y_true = np.argmax(t, axis=1)
    
    acc = float(np.sum(y_pred==y_true)) / len(y_true)
    print('Valid Acc: %.3f' % acc)
    return acc

def test_mnist(x, layers_best):
    y = f_props(layers_best, x)
    y_pred = np.argmax(y, axis=1)

    return y_pred

acc_max = 0
for epoch in range(100):
    x_train, y_train = shuffle(x_train, y_train)
    batch_size = 100
    
    for batch_i in range(batch_size):
        x_batch, y_batch = x_train[batch_i*batch_size:(batch_i+1)*batch_size], y_train[batch_i*batch_size:(batch_i+1)*batch_size]
        train_mnist(x_batch, y_batch, eps=0.1, lam=0.001)
    
    acc = valid_mnist(x_valid, y_valid, layers)
    if acc > acc_max:
        acc_max = acc
        layers_best = copy.deepcopy(layers)
        
y_pred = test_mnist(x_test, layers_best)

submission = pd.Series(y_pred, name='label')
submission.to_csv('./submission_pred.csv', header=True, index_label='id')
