from utils import *
from model import *
from loader import *


def test_loader():
    # loader without labels
    l = Loader(np.ones((1000,1000)))
    batches = list(l.iter_batches())
    for i in range(10):
        b = l.next_batch(100)

    # loader with labels
    l = Loader(np.ones((1000,1000)), np.ones((1000)))
    batches = [b for b,lab in l.iter_batches()]
    for i in range(10):
        b,lab = l.next_batch(100)

def test_saucie():
    data = np.ones((1000,1000))
    labels = np.ones((1000))
    load = Loader(data, labels)

    # saucie with no regularizations
    tf.reset_default_graph()
    saucie = SAUCIE(input_dim = data.shape[1], layer_dimensions=[10,5,2])
    saucie.train(load, steps=10)

    # saucie with c regularization
    tf.reset_default_graph()
    saucie = SAUCIE(input_dim = data.shape[1], layer_dimensions=[10,5,2], lambda_b=.1)
    saucie.train(load, steps=10)

    # saucie with b regularization
    tf.reset_default_graph()
    saucie = SAUCIE(input_dim = data.shape[1], layer_dimensions=[10,5,2], lambda_c=.1)
    saucie.train(load, steps=10)



if __name__=="__main__":
    test_loader()
    test_saucie()
    print("All tests passed.")