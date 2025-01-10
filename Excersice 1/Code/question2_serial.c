from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

if __name__ == '__main__':
    # Generate synthetic classification data
    X, y = make_classification(n_samples=10000, random_state=42, n_features=2, 
                               n_informative=2, n_redundant=0, class_sep=0.8)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Define hyperparameter grid
    params = [{
        'mlp_layer1': [16, 32],
        'mlp_layer2': [16, 32],
        'mlp_layer3': [16, 32]
    }]

    pg = list(ParameterGrid(params))

    # Start timer
    start_time = time.time()
    
    # Serial hyperparameter search
    results = []
    for i, p in enumerate(pg):
        print(p)
        l1 = p['mlp_layer1']
        l2 = p['mlp_layer2']
        l3 = p['mlp_layer3']
        m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        ac = accuracy_score(y_pred, y_test)
        print(i, ac)
        results.append((i, p, ac))
    
    end_time = time.time()
    print(f"Total runtime with serial execution: {end_time - start_time:.2f} seconds")
    
    for r in results:
        print(r)
