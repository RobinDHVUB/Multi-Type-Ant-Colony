# Import to save/load variables into/from a file
import pickle


def store_var(filename, variable):
    """
    Helper function to store a given variable in the given pickle file.
    """
    with open('variables/' + filename + '.pkl','wb') as pf:
        pickle.dump(variable, pf)


def load_var(filename):
    """
    Helper function to load a variable from the given pickle file.
    """
    with open('variables/' + filename + '.pkl','rb') as pf:
        return pickle.load(pf)
