import pandas as pd

# Part 1: Developing the model using the training data ---
print("--- Part 1: Developing the model with training data ---")
try:
    
    train_data = pd.read_csv('train.csv')
    print("Training data loaded successfully!")
except FileNotFoundError:
    print("ERROR: 'train.csv' not found. Please check your file name and location.")
    exit()

# Defining thresholds based on understanding of the data
IMPACT_ACC_THRESHOLD = 27.0
POST_STILLNESS_THRESHOLD = 0.05


# Applying the fall detection logic to the training data
train_data['is_predicted_fall'] = (train_data['acc_max'] > IMPACT_ACC_THRESHOLD) & (train_data['post_lin_max'] < POST_STILLNESS_THRESHOLD)

# Evaluating performance on the training data
actual_falls_train = train_data[train_data['fall'] == 1]
predicted_falls_train = train_data[train_data['is_predicted_fall'] == True]
correct_predictions_train = train_data[(train_data['fall'] == 1) & (train_data['is_predicted_fall'] == True)]
false_alarms_train = train_data[(train_data['fall'] == 0) & (train_data['is_predicted_fall'] == True)]

print(f"\n--- Training Results ---")
print(f"Total actual falls in training data: {len(actual_falls_train)}")
print(f"Total falls predicted by our model: {len(predicted_falls_train)}")
print(f"Correctly identified falls: {len(correct_predictions_train)}")
print(f"False alarms (predicted fall, but was not): {len(false_alarms_train)}")


# Part 2: Evaluating the model on the testing data
print("\n--- Part 2: Evaluating the model on unseen testing data ---")
try:
    test_data = pd.read_csv('test.csv')
    print("Testing data loaded successfully!")
except FileNotFoundError:
    print("ERROR: 'test.csv' not found. Please check your file name and location.")
    exit()

test_data['is_predicted_fall'] = (test_data['acc_max'] > IMPACT_ACC_THRESHOLD) & (test_data['post_lin_max'] < POST_STILLNESS_THRESHOLD)

# Evaluating performance on the testing data
actual_falls_test = test_data[test_data['fall'] == 1]
predicted_falls_test = test_data[test_data['is_predicted_fall'] == True]
correct_predictions_test = test_data[(test_data['fall'] == 1) & (test_data['is_predicted_fall'] == True)]
false_alarms_test = test_data[(test_data['fall'] == 0) & (test_data['is_predicted_fall'] == True)]

print(f"\n--- Testing Results ---")
print(f"Total actual falls in testing data: {len(actual_falls_test)}")
print(f"Total falls predicted on test data: {len(predicted_falls_test)}")
print(f"Correctly identified falls: {len(correct_predictions_test)}")
print(f"False alarms (predicted fall, but was not): {len(false_alarms_test)}")
print("\n--- Program Finished. ---")
