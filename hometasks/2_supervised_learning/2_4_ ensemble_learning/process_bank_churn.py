from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def split_data(raw_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split the data into training and validation sets.

    Args:
    - raw_df (pd.DataFrame): The raw data.
    - test_size (float, optional): The proportion of the data to include in the validation set. Defaults to 0.2.
    - random_state (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
    - train_df (pd.DataFrame): The training data.
    - val_df (pd.DataFrame): The validation data.
    """
    return train_test_split(raw_df, test_size=test_size, random_state=random_state, stratify=raw_df['Exited'])

def get_input_target_cols(raw_df: pd.DataFrame) -> list:
    """
    Get the input column names.

    Args:
    - raw_df (pd.DataFrame): The raw data.

    Returns:
    - input_cols (list): The input column names.
    """
    return raw_df.columns[1:-1].tolist(), raw_df.columns[-1]

def get_numeric_cols(df: pd.DataFrame) -> list:
    """
    Get the numeric column names.

    Args:
    - df (pd.DataFrame): The data.

    Returns:
    - numeric_cols (list): The numeric column names.
    """
    return df.select_dtypes('number').columns.tolist()

def get_categorical_cols(df: pd.DataFrame) -> list:
    """
    Get the categorical column names.

    Args:
    - df (pd.DataFrame): The data.

    Returns:
    - categorical_cols (list): The categorical column names.
    """
    return df.select_dtypes('object').columns.tolist()

def train_encoder(df: pd.DataFrame, cols: list) -> OneHotEncoder:
    """
    Train an OneHotEncoder on the specified columns.

    Args:
    - df (pd.DataFrame): The data.
    - cols (list): The column names to encode.

    Returns:
    - enc (OneHotEncoder): The trained encoder.
    """
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc.fit(df[cols])
    return enc

def transform_with_encoder(df: pd.DataFrame, enc: OneHotEncoder, cols: list) -> pd.DataFrame:
    """
    Transform the data using the OneHotEncoder.

    Args:
    - df (pd.DataFrame): The data.
    - enc (OneHotEncoder): The trained encoder.
    - cols (list): The column names to encode.

    Returns:
    - df (pd.DataFrame): The transformed data.
    """
    encoded_cols = enc.get_feature_names_out(cols)
    df[encoded_cols] = enc.transform(df[cols])
    return df

def train_scaler(df: pd.DataFrame, cols: list) -> StandardScaler:
    """
    Train a StandardScaler on the specified columns.

    Args:
    - df (pd.DataFrame): The data.
    - cols (list): The column names to scale.

    Returns:
    - scaler (StandardScaler): The trained scaler.
    """
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    scaler.fit(df[cols])
    return scaler

def transform_with_scaler(df: pd.DataFrame, scaler: StandardScaler, cols: list) -> pd.DataFrame:
    """
    Transform the data using the StandardScaler.

    Args:
    - df (pd.DataFrame): The data.
    - scaler (StandardScaler): The trained scaler.
    - cols (list): The column names to scale.

    Returns:
    - df (pd.DataFrame): The transformed data.
    """
    df[cols] = scaler.transform(df[cols])
    return df

                        
def split_into_inputs_and_targets(train_df: pd.DataFrame, val_df: pd.DataFrame, input_cols: list, target_col: str) -> tuple:
    """
    Split the data into inputs and targets for training and validation.

    Args:
    - train_df (pd.DataFrame): The training data.
    - val_df (pd.DataFrame): The validation data.
    - input_cols (list): The input column names.
    - target_col (str): The target column name.

    Returns:
    - train_inputs (pd.DataFrame): The training inputs.
    - train_targets (pd.Series): The training targets.
    - val_inputs (pd.DataFrame): The validation inputs.
    - val_targets (pd.Series): The validation targets.
    """
    train_inputs, train_targets = train_df[input_cols], train_df[target_col]
    val_inputs, val_targets = val_df[input_cols], val_df[target_col]

    return train_inputs, train_targets, val_inputs, val_targets     


def preprocess_data(raw_df: pd.DataFrame) -> dict:
    """
    Preprocess the data.

    Args:
    - raw_df (pd.DataFrame): The raw data.

    Returns:
    - result (dict): A dictionary containing the preprocessed data and transformers.
    
    """
    raw_df = raw_df.drop('CustomerId', axis=1)
    train_df, val_df = split_data(raw_df, random_state=247)
    input_cols, target_col = get_input_target_cols(raw_df)

    train_inputs, train_targets, val_inputs, val_targets = split_into_inputs_and_targets(train_df, val_df, input_cols, target_col)

    
    numeric_cols = get_numeric_cols(train_inputs)
    categorical_cols = get_categorical_cols(train_inputs)

    encoder = train_encoder(train_inputs, ['Geography', 'Gender'])
    train_inputs = transform_with_encoder(train_inputs, encoder, ['Geography', 'Gender'])
    val_inputs = transform_with_encoder(val_inputs, encoder, ['Geography', 'Gender'])

    scaler = train_scaler(train_df[input_cols], numeric_cols)
    train_inputs = transform_with_scaler(train_inputs, scaler, numeric_cols)
    val_inputs = transform_with_scaler(val_inputs, scaler, numeric_cols)

    train_inputs = train_inputs.drop(categorical_cols, axis=1)
    val_inputs = val_inputs.drop(categorical_cols, axis=1)

    input_cols = numeric_cols + encoder.get_feature_names_out(['Geography', 'Gender']).tolist()

    return {
        'train_X': train_inputs,
        'train_y': train_targets,
        'val_X': val_inputs,
        'val_y': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }



def preprocess_new_data(test_df: pd.DataFrame, encoder: OneHotEncoder, scaler: StandardScaler) -> tuple:
    """
    Preprocess new data using the already fit encoder and scaler.

    Args:
    - test_df (pd.DataFrame): The new data to preprocess.
    - encoder (OneHotEncoder): The already fit OneHotEncoder.
    - scaler (StandardScaler): The already fit StandardScaler.

    Returns:
    - X_test (pd.DataFrame): The preprocessed new data.
    """
    test_df = test_df.drop('CustomerId', axis=1)
    categorical_cols = get_categorical_cols(test_df)
    numeric_cols = get_numeric_cols(test_df)

    test_df = transform_with_encoder(test_df, encoder, ['Geography', 'Gender'])
    test_df = transform_with_scaler(test_df, scaler, numeric_cols)

    X_test = test_df.drop(categorical_cols, axis=1)

    return X_test






def count_and_plot_auroc(model, inputs, targets, name='', plot=False):

    pred_proba = model.predict_proba(inputs)[:,1]
    fpr, tpr, _ = roc_curve(targets, pred_proba, pos_label=1.)
    roc_auc = auc(fpr, tpr)

    print(f'AUROC for {name}: {roc_auc:.2f}')
        
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve for {name}')
        plt.legend(loc="lower right")
        plt.show()