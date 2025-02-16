import pandas as pd
from sklearn.model_selection import train_test_split

# Assume df is your dataframe
# For instance, you might load your data like this:
df1 = pd.read_csv('Event_occurrence_matrix.csv')

# Step 1: Split the dataframe into training and temporary sets
# Here, 70% of the data goes to training and 30% to the temporary set.
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)

# Step 2: Split the temporary set into validation and test sets
# Here, we split equally, so each gets 15% of the original data.
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Optional: Print out the number of samples in each set
print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")
