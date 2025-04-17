from sklearn.metrics import f1_score
from Thesis_.file_reader import *
from Thesis_.import_file_reader import *

# Calculate the F1 score
#print(predicted)
f1 = f1_score(true, predicted, average='micro')  
print(f"F1 Score: {f1}")
    