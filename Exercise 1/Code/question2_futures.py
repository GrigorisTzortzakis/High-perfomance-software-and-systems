from mpi4py.futures import MPICommExecutor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

def evaluate_parameters(param_set):
    """Evaluate a single set of parameters for the neural network."""
    i, p = param_set
    
    # Generate the same dataset
    X, y = make_classification(n_samples=10000, random_state=42, n_features=2, 
                             n_informative=2, n_redundant=0, class_sep=0.8)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                       random_state=42)
    
    # Extract parameters
    l1 = p['mlp_layer1']
    l2 = p['mlp_layer2']
    l3 = p['mlp_layer3']
    
    # Create and train the model
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3), random_state=42)
    m.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    
    return i, p, ac

def main():
    start_time = time.time()
    
    # Define parameter grid
    params = [{'mlp_layer1': [16, 32],
               'mlp_layer2': [16, 32],
               'mlp_layer3': [16, 32]}]
    
    pg = ParameterGrid(params)
    param_sets = list(enumerate(pg))
    
    # Using MPI futures
    with MPICommExecutor() as executor:
        if executor is not None:  # Main process
            comp_start_time = time.time()
            
            # Map parameter sets to worker processes
            results = list(executor.map(evaluate_parameters, param_sets))
            
            comp_time = time.time() - comp_start_time
            total_time = time.time() - start_time
            
            # Sort results by index
            results.sort(key=lambda x: x[0])
            
            # Print results and timing
            print("\nResults:")
            for r in results:
                print(r)
                
            print(f"\nTiming Information:")
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Computation time: {comp_time:.2f} seconds")

if __name__ == '__main__':
    main()