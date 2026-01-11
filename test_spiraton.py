import numpy as np

from spiraton import Spiraton, SpiralGrid, Tokenizer8Bit


def test_spiral_grid_propagate_length():
    np.random.seed(0)
    grid = SpiralGrid(num_units=3, input_size=3)
    inputs = np.array([0.2, -0.1, 0.5])
    outputs = grid.propagate(inputs)
    assert len(outputs) == 3


def test_spiraton_training_reduces_error():
    np.random.seed(0)
    spir = Spiraton(3)
    inputs = np.array([0.5, -0.2, 0.1])
    target = 0.05
    before = spir.operate(inputs)
    for _ in range(10):
        spir.train(inputs, target, learning_rate=0.05)
    after = spir.operate(inputs)
    assert abs(target - after) < abs(target - before)


def test_cycle_closed_updates_intention_and_memory():
    np.random.seed(1)
    spir = Spiraton(3)
    inputs = np.array([0.1, 0.2, -0.3])
    state = spir.cycle(inputs, intention=0.4, learning_rate=0.05, closed_loop=True)
    assert state.closed_loop is True
    assert state.alpha_prime != state.alpha
    assert len(spir.memory) == 1


def test_cycle_open_does_not_update_parameters():
    np.random.seed(2)
    spir = Spiraton(3)
    inputs = np.array([-0.2, 0.3, 0.1])
    weights_before = spir.weights.copy()
    bias_before = spir.bias
    spir.cycle(inputs, intention=-0.1, learning_rate=0.05, closed_loop=False)
    assert np.allclose(weights_before, spir.weights)
    assert bias_before == spir.bias


def test_tokenizer8bit_counts_and_encode():
    tokenizer = Tokenizer8Bit.from_corpus("abcabc")
    tokens = tokenizer.encode("abc")
    assert tokens.dtype == np.uint8
    assert tokenizer.counts.sum() == 6


def test_tokenizer8bit_vectorize_shape():
    tokenizer = Tokenizer8Bit.from_corpus("spiral")
    vector = tokenizer.vectorize("spiral", input_size=4)
    assert vector.shape == (4,)
