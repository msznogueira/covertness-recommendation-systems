import scipy.io
import pandas as pd

# Load the .mat file
fnames = ["lastfm_U_matrix", "movielens1k_U_matrix"]
for fname in fnames:
    mat_data = scipy.io.loadmat("./" + fname + ".mat")

    # Extract your specific variable name (replace 'variable_name')
    data_array = mat_data['u']

    # Convert to a Pandas DataFrame and save as CSV
    df = pd.DataFrame(data_array)
    df.to_csv(fname + '.csv', index=False)
