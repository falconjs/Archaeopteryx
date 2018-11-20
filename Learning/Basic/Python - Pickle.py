import os
import pickle

example_dict = {1:"d", 2:"2", 3:"f"}

root_dir = "./Archaeopteryx/Learning/Basic"
pickle_file = os.path.join(root_dir, 'dict.pickle')

# ===================== Write ======================

pickle_out = open(pickle_file, "wb")
pickle.dump(example_dict, pickle_out)
pickle_out.close()


pickle_in = open(pickle_file, "rb")
example_dict2 = pickle.load(pickle_in)
pickle_in.close()


example_dt1 = {1:"d", 2:"e", 3:"f"}
example_dt2 = {4:"s", 5:"d", 6:"g"}
example_dt3 = {7:"u", 8:"i", 9:"o"}

f = open(pickle_file, 'wb')
save = {
    'example_dt1': example_dt1,
    'example_dt2': example_dt2,
    'example_dt3': example_dt3,
}
pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
f.close()

# ====================== Read =======================

pickle_in = open(pickle_file, "rb")
example_dict2 = pickle.load(pickle_in)
pickle_in.close()

example_dt4 = example_dict2['example_dt1']


# try
# except Exception as e:
#     print('Unable to save data to', pickle_file, ':', e)
#     raise