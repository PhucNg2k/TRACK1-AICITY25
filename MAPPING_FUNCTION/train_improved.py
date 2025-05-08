import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

# 1. Load and clean data
def load_and_clean_data():
    try:
        df = pd.read_csv('out.csv')
        print("Data loaded successfully from out.csv")

        unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)
            print(f"Dropped 'Unnamed' columns: {unnamed_cols}")
        else:
            print("No 'Unnamed' columns found.")

        return df

    except FileNotFoundError:
        print("Error: out.csv not found. Please make sure the file is in the correct directory.")
        exit()

# 2. Data quality check function
def check_data_quality(df):
    print("\nData Quality Report:")
    print("-" * 50)
    
    # Check for missing values
    print("Missing values:\n", df.isnull().sum())
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    print("\nInfinite values:\n", np.isinf(df[numeric_cols]).sum())
    
    # Basic statistics
    print("\nDataset shape:", df.shape)
    print("\nNumber of unique cameras:", df['cam_id'].nunique())
    print("Number of unique objects:", df['obj_id'].nunique())
    
    return df

# 3. Parse list strings
def parse_list_string(list_str):
    try:
        return np.fromstring(list_str.strip('[]'), sep=',')
    except:
        return np.nan

def parse_list_columns(df):
    cols_to_parse = ['cam_pos_3d', 'cam_3d_direct', '3d_loc', '2d_visible', 
                     'map_2d', 'reproject_pix', 'cam_2d_pos']
    
    for col in cols_to_parse:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_string)
        else:
            print(f"Warning: Column '{col}' not found in the CSV.")
    return df

# 4. Expand coordinate columns
def expand_coordinate_columns(df, col_name, num_dims, new_col_prefix):
    if col_name in df.columns and df[col_name].iloc[0] is not np.nan:
        try:
            expanded_data = pd.DataFrame(df[col_name].tolist(), index=df.index)
            new_cols = [f'{new_col_prefix}_{i+1}' for i in range(num_dims)]
            if expanded_data.shape[1] == num_dims:
                df[new_cols] = expanded_data
                df.drop(columns=[col_name], inplace=True)
                print(f"Column '{col_name}' expanded into: {new_cols}")
            else:
                print(f"Warning: Column '{col_name}' does not consistently have {num_dims} dimensions.")
        except Exception as e:
            print(f"Error expanding column '{col_name}': {e}")
    return df

def expand_all_coordinates(df):
    df = expand_coordinate_columns(df, 'cam_pos_3d', 2, 'cam_pos')
    df = expand_coordinate_columns(df, 'cam_3d_direct', 3, 'cam_3d_direct')
    df = expand_coordinate_columns(df, '2d_visible', 2, '2d_visible')
    df = expand_coordinate_columns(df, '3d_loc', 3, '3d_loc')
    df = expand_coordinate_columns(df, 'map_2d', 2, 'map_2d')
    df = expand_coordinate_columns(df, 'reproject_pix', 2, 'reproject_pix')
    df = expand_coordinate_columns(df, 'cam_2d_pos', 2, 'cam_2d_pos')
    return df

# 5. Add derived features
def add_derived_features(df):
    # Normalized 2D coordinates
    df['2d_visible_1_norm'] = df['2d_visible_1'] / df['cam_frameWidth']
    df['2d_visible_2_norm'] = df['2d_visible_2'] / df['cam_frameHeight']
    
    # Distance from image center
    df['dist_from_center_x'] = df['2d_visible_1'] - (df['cam_frameWidth'] / 2)
    df['dist_from_center_y'] = df['2d_visible_2'] - (df['cam_frameHeight'] / 2)
    
    # Euclidean distance from image center
    df['dist_from_center'] = np.sqrt(df['dist_from_center_x']**2 + df['dist_from_center_y']**2)
    
    print("Added derived features.")
    return df

# 6. Feature selection
def select_features(df):
    # Camera parameters
    camera_features = ['cam_pos_1', 'cam_pos_2', 'cam_2d_direct',
                      'cam_3d_direct_1', 'cam_3d_direct_2', 'cam_3d_direct_3']
    
    # 2D coordinates and derived features
    coord_features = ['2d_visible_1', '2d_visible_2',
                     '2d_visible_1_norm', '2d_visible_2_norm',
                     'dist_from_center_x', 'dist_from_center_y',
                     'dist_from_center']
    
    # Additional geometric features
    geometric_features = ['map_2d_1', 'map_2d_2',
                         'reproject_pix_1', 'reproject_pix_2',
                         'cam_2d_pos_1', 'cam_2d_pos_2']
    
    all_features = camera_features + coord_features + geometric_features
    return df[all_features]

# 7. Create preprocessing pipeline
def create_preprocessing_pipeline():
    # Define feature groups
    camera_features = ['cam_pos_1', 'cam_pos_2', 'cam_2d_direct',
                      'cam_3d_direct_1', 'cam_3d_direct_2', 'cam_3d_direct_3']
    
    coord_features = ['2d_visible_1', '2d_visible_2']
    
    derived_features = ['2d_visible_1_norm', '2d_visible_2_norm',
                       'dist_from_center_x', 'dist_from_center_y',
                       'dist_from_center']
    
    geometric_features = ['map_2d_1', 'map_2d_2',
                         'reproject_pix_1', 'reproject_pix_2',
                         'cam_2d_pos_1', 'cam_2d_pos_2']
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('camera_scaler', RobustScaler(), camera_features),
            ('coord_scaler', StandardScaler(), coord_features),
            ('derived_scaler', StandardScaler(), derived_features),
            ('geometric_scaler', RobustScaler(), geometric_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

'''
Why Split by Object ID?
    Each object (person/vehicle) has multiple observations
    We want ALL observations of the same object to stay together
    This prevents data leakage (same object appearing in different sets)
'''


# 8. Split data
def split_data(X, y, df):
    # Get unique object IDs for splitting
    unique_objects = df['obj_id'].unique()
    
    # Split objects into train, validation, and test sets
    train_objects, temp_objects = train_test_split(
        unique_objects, 
        test_size=0.3,  # 30% for validation + test
        random_state=42
    )
    
    valid_objects, test_objects = train_test_split(
        temp_objects, 
        test_size=0.5,  # 50% of the 30% for test (15% total)
        random_state=42
    )
    
    # Create masks for splits
    train_mask = df['obj_id'].isin(train_objects)
    valid_mask = df['obj_id'].isin(valid_objects)
    test_mask = df['obj_id'].isin(test_objects)
    
    # Split the data
    X_train = X[train_mask]
    X_valid = X[valid_mask]
    X_test = X[test_mask]
    
    y_train = y[train_mask]
    y_valid = y[valid_mask]
    y_test = y[test_mask]
    
    # Print split statistics
    print("\nData Split Statistics:")
    print("-" * 50)
    print(f"Training set size: {len(X_train)} ({len(train_objects)} unique objects)")
    print(f"Validation set size: {len(X_valid)} ({len(valid_objects)} unique objects)")
    print(f"Test set size: {len(X_test)} ({len(test_objects)} unique objects)")
    
    # Verify no overlap between splits
    train_valid_overlap = set(train_objects).intersection(set(valid_objects))
    train_test_overlap = set(train_objects).intersection(set(test_objects))
    valid_test_overlap = set(valid_objects).intersection(set(test_objects))
    
    if not (train_valid_overlap or train_test_overlap or valid_test_overlap):
        print("\nSplit verification: No overlap between splits ✓")
    else:
        print("\nWarning: Overlap detected between splits!")
        if train_valid_overlap:
            print(f"Train-Validation overlap: {len(train_valid_overlap)} objects")
        if train_test_overlap:
            print(f"Train-Test overlap: {len(train_test_overlap)} objects")
        if valid_test_overlap:
            print(f"Validation-Test overlap: {len(valid_test_overlap)} objects")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# 9. Create model
def create_model():
    base_regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    
    model = MultiOutputRegressor(base_regressor)
    return model

# 9.1 Create PyTorch Neural Network
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TwoDtoThreeDModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=3, dropout_rate=0.3):
        super(TwoDtoThreeDModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_pytorch_model(X_train, y_train, X_valid, y_valid, input_dim, 
                       batch_size=256, n_epochs=100, learning_rate=0.001):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_valid_tensor = torch.FloatTensor(X_valid.values)
    y_valid_tensor = torch.FloatTensor(y_valid.values)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    model = TwoDtoThreeDModel(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_valid_loss = float('inf')
    best_model_state = None
    
    print("\nTraining PyTorch Neural Network:")
    print("-" * 50)
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(n_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
        for X_batch, y_batch in train_pbar:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Validation phase
        model.eval()
        valid_loss = 0
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Valid]", leave=False)
        with torch.no_grad():
            for X_batch, y_batch in valid_pbar:
                y_pred = model(X_batch)
                valid_loss += criterion(y_pred, y_batch).item()
                valid_pbar.set_postfix({'loss': f'{valid_loss/len(valid_loader):.6f}'})
        
        # Calculate average losses
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        
        # Learning rate scheduling
        scheduler.step(valid_loss)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict().copy()
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'valid_loss': f'{valid_loss:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # Load best model
    model.load_state_dict(best_model_state)
    print("\nTraining completed!")
    print(f"Best validation loss: {best_valid_loss:.6f}")
    
    return model

def train_randomforest_pipeline(X_train, y_train, X_valid, y_valid):
    """Train RandomForest model pipeline with progress tracking"""
    print("\nTraining RandomForest model...")
    print("-" * 50)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', create_model())
    ])
    
    # Train the model with progress bar
    with tqdm(total=1, desc="Training RandomForest") as pbar:
        pipeline.fit(X_train, y_train)
        pbar.update(1)
    
    print("RandomForest training completed!")
    return pipeline

def train_pytorch_pipeline(X_train, y_train, X_valid, y_valid):
    """Train PyTorch model pipeline with progress tracking"""
    input_dim = X_train.shape[1]  # Number of features
    return train_pytorch_model(X_train, y_train, X_valid, y_valid, input_dim)

def evaluate_randomforest_model(pipeline, X_valid, y_valid, X_test, y_test):
    """Evaluate RandomForest model"""
    print("\nEvaluating RandomForest model:")
    y_valid_pred_rf = pipeline.predict(X_valid)
    valid_mse_rf, valid_errors_rf = evaluate_model(y_valid, y_valid_pred_rf, "validation (RF)")
    
    y_test_pred_rf = pipeline.predict(X_test)
    test_mse_rf, test_errors_rf = evaluate_model(y_test, y_test_pred_rf, "test (RF)")
    
    return valid_mse_rf, valid_errors_rf, test_mse_rf, test_errors_rf

def evaluate_pytorch_model(model, X_valid, y_valid, X_test, y_test):
    """Evaluate PyTorch model"""
    print("\nEvaluating PyTorch model:")
    model.eval()
    with torch.no_grad():
        X_valid_tensor = torch.FloatTensor(X_valid.values)
        X_test_tensor = torch.FloatTensor(X_test.values)
        
        y_valid_pred_nn = model(X_valid_tensor).numpy()
        y_test_pred_nn = model(X_test_tensor).numpy()
    
    valid_mse_nn, valid_errors_nn = evaluate_model(y_valid, y_valid_pred_nn, "validation (NN)")
    test_mse_nn, test_errors_nn = evaluate_model(y_test, y_test_pred_nn, "test (NN)")
    
    return valid_mse_nn, valid_errors_nn, test_mse_nn, test_errors_nn

# 10. Model evaluation
def evaluate_model(y_true, y_pred, set_name=""):
    print(f"\nEvaluation on {set_name} set:")
    print("-" * 50)
    
    # MSE per coordinate
    mse_per_coord = np.mean((y_true - y_pred)**2, axis=0)
    print("MSE per coordinate:")
    print(f"X: {mse_per_coord.iloc[0]:.6f}")
    print(f"Y: {mse_per_coord.iloc[1]:.6f}")
    print(f"Z: {mse_per_coord.iloc[2]:.6f}")
    
    # Euclidean distance error
    euclidean_errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
    
    print(f"\nError Statistics:")
    print(f"Mean Euclidean Error: {np.mean(euclidean_errors):.6f}")
    print(f"Median Euclidean Error: {np.median(euclidean_errors):.6f}")
    print(f"95th percentile Error: {np.percentile(euclidean_errors, 95):.6f}")
    
    # Additional statistics
    print(f"\nCoordinate-wise Statistics:")
    for i, coord in enumerate(['X', 'Y', 'Z']):
        errors = np.abs(y_true.iloc[:, i] - y_pred.iloc[:, i])
        print(f"{coord} coordinate:")
        print(f"  Mean Absolute Error: {np.mean(errors):.6f}")
        print(f"  Median Absolute Error: {np.median(errors):.6f}")
        print(f"  95th percentile Error: {np.percentile(errors, 95):.6f}")
    
    return mse_per_coord, euclidean_errors

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_clean_data()
    df = check_data_quality(df)
    df = parse_list_columns(df)
    df = expand_all_coordinates(df)
    df = add_derived_features(df)
    
    # Prepare features and target
    X = select_features(df)
    y = df[['3d_loc_1', '3d_loc_2', '3d_loc_3']]
    
    # Split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y, df)
    
    # Train models (comment out the ones you don't want to train)
    rf_pipeline = train_randomforest_pipeline(X_train, y_train, X_valid, y_valid)
    #pytorch_model = train_pytorch_pipeline(X_train, y_train, X_valid, y_valid)
    
    # Evaluate models (comment out the ones you don't want to evaluate)
    valid_mse_rf, valid_errors_rf, test_mse_rf, test_errors_rf = evaluate_randomforest_model(
        rf_pipeline, X_valid, y_valid, X_test, y_test
    )
    
    #valid_mse_nn, valid_errors_nn, test_mse_nn, test_errors_nn = evaluate_pytorch_model(
    #    pytorch_model, X_valid, y_valid, X_test, y_test
    #) 