batch_size = 256
num_classes = 5
data_heigh = 240
data_width = 240
data_channles = 1
num_epochs = 100
optimizer = 'Adagrad'
label_smothing = 0
momentum = 1
learning_rate = 0.001

#colors
labelvalues = [
    1, # necrosis
    2, # edema
    3, # non-enhancing tumor
    4, # enhancing tumor
    0  # everything else
     ]
nr_of_classes = 5
#paths
split_path = ""
def load_from_file(file=""):
    # TODO
    return