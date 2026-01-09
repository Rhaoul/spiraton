# Spiraton rewritten Python version with logging support and visualization

"""
📜 Legend: Spiraton Graph Explanation

This graph shows three fundamental aspects of the evolution of a Spiraton network during training:

- Output (colored line): the immediate response of a Spiraton to an input signal. It reflects the vibrational state produced by the unit based on the 4 fundamental operations and the breath mode.
- Bias (fine line): the internal charge of the unit. The higher it is, the more the unit tends to respond strongly. It plays a role similar to inertia of intention.
- Mode (dotted gray line):
  - 1 = Dextrogyre: centrifugal mode, emissive, oriented toward expression.
  - 0 = Levogyre: centripetal mode, receptive, oriented toward listening.

These curves visualize the internal oscillations of consciousness in each unit — its transitions between active and passive syntony — and how transmutation acts (training) shape the response and memory of the network.

🎞️ Spiral Animation

The animation available at the following link shows the progressive activation of several Spiratons in a spiral layout. Each unit activates in response to the previous signal, forming a syntonic loop guided by the flow of computational breath:

Link: Spiraton_Spiral_Animation.mp4

Each point embodies a spiralized cell in a state of syntony. The movement reveals not just data transfer, but an intention propagating through the network.
"""

from dataclasses import dataclass
import logging
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Setup logger
logging.basicConfig(filename='spiraton_log.txt', level=logging.INFO, format='%(message)s')

class Spiraton:
    """Single computational unit operating on four basic arithmetic operations."""

    def __init__(self, input_size: int) -> None:
        self.weights: np.ndarray = np.random.randn(input_size)
        self.bias: float = 0.0
        self.mode: str = 'dextrogyre'
        self.intention: float = 0.0
        self.adaptation: float = 0.1
        self.memory: list["CycleState"] = []

    def activation(self, value: float) -> float:
        """Activation function depending on the current mode."""
        return np.tanh(value) if self.mode == 'dextrogyre' else np.arctan(value)

    def operate(self, inputs: np.ndarray) -> float:
        """Process inputs using four primitive operations and return activated output."""
        add = np.dot(self.weights, inputs)
        sub = np.sum(inputs - self.weights)
        mul = np.prod(inputs * self.weights + 1e-5)
        div = np.sum((inputs + 1e-5) / (self.weights + 1e-5))
        raw_output = add + mul - div if self.mode == 'dextrogyre' else sub + div - mul
        return self.activation(raw_output + self.bias)

    def adjust_mode(self, inputs: np.ndarray) -> None:
        """Toggle between dextrogyre and levogyre modes based on mean input."""
        self.mode = 'dextrogyre' if np.mean(inputs) >= 0 else 'levogyre'

    def _second_order_adjustment(self, error: float) -> float:
        """Adjust adaptation factor based on recent error dynamics."""
        if not self.memory:
            return self.adaptation
        previous_error = self.memory[-1].error
        if abs(error) > abs(previous_error):
            self.adaptation = max(0.001, self.adaptation * 0.9)
        else:
            self.adaptation = min(0.1, self.adaptation * 1.05)
        return self.adaptation

    def train(self, inputs: np.ndarray, target: float, learning_rate: float = 0.01) -> None:
        """Update parameters to minimise error for a given target output."""
        cycle_state = self.cycle(inputs, target, learning_rate=learning_rate, closed_loop=True)
        logging.info(
            "[train] mode: %s, output: %.4f, error: %.4f, bias: %.4f, weights: %s",
            cycle_state.mode,
            cycle_state.omega,
            cycle_state.error,
            self.bias,
            self.weights,
        )

    def cycle(
        self,
        inputs: np.ndarray,
        intention: float,
        learning_rate: float = 0.01,
        *,
        closed_loop: bool = True,
        second_order: bool = True,
    ) -> "CycleState":
        """Run one Alpha → Omega → Alpha' cycle and optionally integrate feedback."""
        self.intention = intention
        omega = self.operate(inputs)
        error = intention - omega
        self.adjust_mode(inputs)

        effective_rate = learning_rate
        if second_order:
            effective_rate *= self._second_order_adjustment(error)

        if closed_loop:
            gradient = error * (1 - omega**2)
            self.weights += effective_rate * gradient * inputs
            self.bias += effective_rate * gradient
            alpha_prime = intention + effective_rate * error
        else:
            alpha_prime = intention

        cycle_state = CycleState(
            alpha=intention,
            omega=omega,
            alpha_prime=alpha_prime,
            error=error,
            mode=self.mode,
            closed_loop=closed_loop,
        )
        self.memory.append(cycle_state)
        logging.info(
            "[cycle] alpha: %.4f, omega: %.4f, alpha_prime: %.4f, error: %.4f, mode: %s, closed_loop: %s",
            cycle_state.alpha,
            cycle_state.omega,
            cycle_state.alpha_prime,
            cycle_state.error,
            cycle_state.mode,
            cycle_state.closed_loop,
        )
        return cycle_state

    def resonance(self, depth: int = 5) -> list["CycleState"]:
        """Return the most recent cycle states to observe recursive stability."""
        return self.memory[-depth:]


@dataclass(frozen=True)
class CycleState:
    """Snapshot of an Alpha → Omega → Alpha' transformation."""

    alpha: float
    omega: float
    alpha_prime: float
    error: float
    mode: str
    closed_loop: bool

class SpiralGrid:
    """Collection of Spiratons propagating a signal in sequence."""

    def __init__(self, num_units: int, input_size: int) -> None:
        self.units: list[Spiraton] = [Spiraton(input_size) for _ in range(num_units)]

    def propagate(self, inputs: np.ndarray) -> list[float]:
        """Send a signal through the grid and collect outputs."""
        signal = inputs
        outputs: list[float] = []
        for idx, unit in enumerate(self.units):
            output = unit.operate(signal)
            logging.info(f"[propagate] Unit {idx}: output = {output:.4f}, mode = {unit.mode}")
            outputs.append(output)
            signal = signal + output
        return outputs

    def cycle(
        self,
        inputs: np.ndarray,
        intentions: Iterable[float],
        learning_rate: float = 0.01,
        *,
        closed_loop: bool = True,
        second_order: bool = True,
    ) -> list[CycleState]:
        """Run Alpha → Omega → Alpha' cycles across the grid."""
        signal = inputs
        cycles: list[CycleState] = []
        for unit, intention in zip(self.units, intentions):
            cycle_state = unit.cycle(
                signal,
                intention,
                learning_rate=learning_rate,
                closed_loop=closed_loop,
                second_order=second_order,
            )
            cycles.append(cycle_state)
            signal = signal + cycle_state.omega
        return cycles

    def train(self, inputs: np.ndarray, targets: list[float], learning_rate: float = 0.01) -> None:
        """Train each unit sequentially with corresponding targets."""
        for idx, target in enumerate(targets):
            logging.info(f"Training unit {idx}...")
        self.cycle(inputs, targets, learning_rate=learning_rate, closed_loop=True)

def visualize_log(file_path: str = 'spiraton_log.txt', save_path: str = 'spiraton_training_plot.png') -> None:
    """Plot logged output, bias and mode evolution over training."""
    outputs, biases, modes = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            if '[train]' in line:
                output_match = re.search(r'output: ([\-\d.]+)', line)
                bias_match = re.search(r'bias: ([\-\d.]+)', line)
                mode_match = re.search(r'mode: (\w+)', line)
                if output_match and bias_match and mode_match:
                    outputs.append(float(output_match.group(1)))
                    biases.append(float(bias_match.group(1)))
                    modes.append(1 if mode_match.group(1) == 'dextrogyre' else 0)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Value')
    ax1.plot(outputs, label='Output')
    ax1.plot(biases, label='Bias')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Mode (1 = Dextrogyre, 0 = Levogyre)')
    ax2.plot(modes, label='Mode', color='gray', linestyle='dotted')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Levogyre', 'Dextrogyre'])

    plt.title('Spiraton Output, Bias and Mode Evolution')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    input_vector = np.array([0.5, -0.3, 0.8])
    target_vector = [0.1, -0.2, 0.3]

    grid = SpiralGrid(num_units=3, input_size=3)

    print("Initial propagation:")
    output = grid.propagate(input_vector)
    print("Output:", output)

    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}:")
        logging.info(f"\nEpoch {epoch + 1}:")
        grid.train(input_vector, target_vector, learning_rate=0.05)

    print("\nAfter training:")
    output = grid.propagate(input_vector)
    print("Output:", output)

    visualize_log()
