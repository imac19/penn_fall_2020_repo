import numpy as np


# When you turn this function in to Gradescope, it is easiest to copy and paste this cell to a new python file called hw1.py
# and upload that file instead of the full Jupyter Notebook code (which will cause problems for Gradescope)
def compute_features(names):
    
    feature_array = np.zeros(shape=(len(names), 260), dtype=int)
    
    for i in range (0, len(names)):
        
        name_split = names[i].lower().split()
        
        for j in range (0,2):     
            
            for k in range (0, min(5, len(name_split[j]))):   
                
                buffer = (26*k) + (130*j) - 97
                position = ord(name_split[j][k]) + buffer
                feature_array[i][position] = 1
                
    
    return feature_array     