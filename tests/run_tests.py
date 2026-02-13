import unittest
import sys
import os

if __name__ == "__main__":
    # Ensure the project root is in the path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
