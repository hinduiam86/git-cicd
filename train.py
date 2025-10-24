import pickle
import cProfile
import pstats
import io
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

def train_model():
    # Load dataset
    iris = load_iris()
    clf = RandomForestClassifier()
    clf.fit(iris.data, iris.target)

    # Save the trained model
    os.makedirs("model", exist_ok=True)
    with open("model/my_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Model saved to model/my_model.pkl")

if __name__ == "__main__":
    # Profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    train_model()
    
    profiler.disable()
    
    # Save profiling result
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    ps.print_stats()
    
    with open("model/profile.txt", "w") as f:
        f.write(s.getvalue())
    
    print("Profiling saved to model/profile.txt")