import numpy as np
from spiraton import Spiraton, SpiralGrid


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
