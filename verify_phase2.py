import sys
import os
sys.path.insert(0, os.path.abspath("."))

from tests.test_core_cell import test_matrix_core_shapes_and_finiteness, test_matrix_cell_non_commutative
from tests.test_alpha_omega_spatial import test_double_dynamic_asymmetry

def run():
    print("Testing Matrix shapes...")
    test_matrix_core_shapes_and_finiteness(8, 4)
    print("Testing Commutator...")
    test_matrix_cell_non_commutative()
    print("Testing Double Dynamic...")
    test_double_dynamic_asymmetry()
    print("All Phase 2 tests passed!")

if __name__ == "__main__":
    run()
