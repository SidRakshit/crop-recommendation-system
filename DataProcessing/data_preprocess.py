import pandas as pd

data_file_name = '../Datasets/Crop_recommendation.csv'
data = pd.read_csv(data_file_name)

data['NPK_sum'] = data['N'] + data['P'] + data['K']

data['NPK_mean'] = data[['N', 'P', 'K']].mean(axis=1)

data['NPK_weighted'] = 0.5 * data['N'] + 0.3 * data['P'] + 0.2 * data['K']

output_file_name = 'Modified_Crop_Recommendation.csv'
data.to_csv(output_file_name, index=False)

print(f"DataFrame has been successfully saved to {output_file_name}")
