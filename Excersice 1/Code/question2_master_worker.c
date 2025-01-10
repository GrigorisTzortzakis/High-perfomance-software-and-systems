from mpi4py import MPI
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluate_params(p, X_train, X_test, y_train, y_test):
    """
    Train and evaluate the MLPClassifier with given hyperparameters.

    Parameters:
        p (dict): Hyperparameter combination.
        X_train, X_test, y_train, y_test: Training and testing datasets.

    Returns:
        tuple: (hyperparameters, accuracy)
    """
    try:
        clf = MLPClassifier(
            hidden_layer_sizes=(p['mlp_layer1'], p['mlp_layer2'], p['mlp_layer3']),
            random_state=42,
            max_iter=200  # Adjust as needed
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return (p, acc)
    except Exception as e:
        return (p, str(e))

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Master process (rank {rank}) starting.")
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=10000,
            random_state=42,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            class_sep=0.8
        )
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        
        # Define hyperparameter grid
        params = [{
            'mlp_layer1': [16, 32],
            'mlp_layer2': [16, 32],
            'mlp_layer3': [16, 32]
        }]
        pg = list(ParameterGrid(params))
        num_tasks = len(pg)
        results = []
        print(f"Total tasks: {num_tasks}")
        
        # Initialize task index
        task_index = 0
        
        # Distribute initial tasks to workers
        for worker in range(1, min(size, num_tasks + 1)):
            p = pg[task_index]
            comm.send((p, X_train, X_test, y_train, y_test), dest=worker, tag=11)
            print(f"Master sent task {task_index} to worker {worker}")
            task_index += 1
        
        # Assign remaining tasks dynamically
        for _ in range(num_tasks):
            # Receive result from any worker
            status = MPI.Status()
            result = comm.recv(source=MPI.ANY_SOURCE, tag=22, status=status)
            source = status.Get_source()
            results.append(result)
            print(f"Master received result from worker {source}: {result}")
            
            # Assign next task if available
            if task_index < num_tasks:
                p = pg[task_index]
                comm.send((p, X_train, X_test, y_train, y_test), dest=source, tag=11)
                print(f"Master sent task {task_index} to worker {source}")
                task_index += 1
            else:
                # No more tasks; send termination signal
                comm.send(None, dest=source, tag=11)
                print(f"Master sent termination signal to worker {source}")
        
        # After all tasks are completed, print the results
        print("\nAll tasks completed. Printing results:")
        for idx, (p, acc) in enumerate(results):
            print(f"Run {idx}: Parameters: {p}, Accuracy: {acc}")
    
    else:
        # Worker Processes
        while True:
            data = comm.recv(source=0, tag=11)
            if data is None:
                # Termination signal received
                print(f"Worker {rank} received termination signal.")
                break
            p, X_train, X_test, y_train, y_test = data
            print(f"Worker {rank} received task: {p}")
            result = evaluate_params(p, X_train, X_test, y_train, y_test)
            comm.send(result, dest=0, tag=22)
            print(f"Worker {rank} completed task: {p}")

if __name__ == '__main__':
    main()
