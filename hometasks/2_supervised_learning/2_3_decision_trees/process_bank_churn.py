


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

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

def get_input_cols(raw_df: pd.DataFrame) -> list:
    """
    Get the input column names.

    Args:
    - raw_df (pd.DataFrame): The raw data.

    Returns:
    - input_cols (list): The input column names.
    """
    return raw_df.columns[1:-1].tolist()

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
    scaler = StandardScaler()
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

                        


def preprocess_data(raw_df: pd.DataFrame) -> dict:
    """
    Preprocess the data.

    Args:
    - raw_df (pd.DataFrame): The raw data.

    Returns:
    - result (dict): A dictionary containing the preprocessed data and transformers.
    """
    train_df, val_df = split_data(raw_df)
    input_cols = get_input_cols(raw_df)
    numeric_cols = get_numeric_cols(train_df)
    categorical_cols = get_categorical_cols(train_df)

    encoder = train_encoder(train_df, ['Geography', 'Gender'])
    train_df = transform_with_encoder(train_df, encoder, ['Geography', 'Gender'])
    val_df = transform_with_encoder(val_df, encoder, ['Geography', 'Gender'])

    scaler = train_scaler(train_df, numeric_cols)
    train_df = transform_with_scaler(train_df, scaler, numeric_cols)
    val_df = transform_with_scaler(val_df, scaler, numeric_cols)

    X_train = train_df.drop(categorical_cols, axis=1)
    X_val = val_df.drop(categorical_cols, axis=1)

    input_cols = numeric_cols + encoder.get_feature_names_out(['Geography', 'Gender']).tolist()

    return {
        'train_X': X_train,
        'train_y': train_df['Exited'],
        'val_X': X_val,
        'val_y': val_df['Exited'],
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }



def preprocess_new_data(test_df: pd.DataFrame, encoder: OneHotEncoder, scaler: StandardScaler, input_cols: list) -> tuple:
    """
    Preprocess new data using the already fit encoder and scaler.

    Args:
    - test_df (pd.DataFrame): The new data to preprocess.
    - encoder (OneHotEncoder): The already fit OneHotEncoder.
    - scaler (StandardScaler): The already fit StandardScaler.
    - input_cols (list): The input column names.

    Returns:
    - X_test (pd.DataFrame): The preprocessed new data.
    """
    categorical_cols = get_categorical_cols(test_df)
    numeric_cols = get_numeric_cols(test_df)

    test_df = transform_with_encoder(test_df, encoder, ['Geography', 'Gender'])
    test_df = transform_with_scaler(test_df, scaler, numeric_cols)

    X_test = test_df.drop(categorical_cols, axis=1)

    return X_test




# def preprocess_data(raw_df):


#     train_df, val_df = train_test_split(raw_df, random_state=42, train_size=0.2, stratify=raw_df['Exited'])

#     # Створюємо трен. і вал. набори
#     input_cols = raw_df.columns[1:-1].tolist()
#     target_col = raw_df.columns[-1]
#     train_inputs, train_targets = train_df[input_cols], train_df[target_col]
#     val_inputs, val_targets = val_df[input_cols], val_df[target_col]

#     # Виявляємо числові і категоріальні колонки
#     numeric_cols = train_inputs.select_dtypes('number').columns.tolist()
#     categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


#     # тренуємо енкодер на тренувальних даних
#     enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     enc.fit(train_inputs[['Geography', 'Gender']])
#     encoded_cols = enc.get_feature_names_out(['Geography', 'Gender'])
#     train_inputs[encoded_cols] = enc.transform(train_inputs[['Geography', 'Gender']])
#     val_inputs[encoded_cols] = enc.transform(val_inputs[['Geography', 'Gender']])
#     # train_inputs = train_inputs.drop(['Gender_Female'], axis=1)
#     # val_inputs = val_inputs.drop(['Gender_Female'], axis=1)


#     scaler = StandardScaler()
#     # натренуємо скейлер на тренувальних даних
#     scaler.fit(train_inputs[numeric_cols])

#     # трансформуємо дані
#     train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
#     val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

#     # train_inputs = train_inputs.drop(['CustomerId'], axis=1)
#     # val_inputs = val_inputs.drop(['CustomerId'], axis=1)
#     X_train = train_inputs.drop(categorical_cols, axis=1)
#     X_val =  val_inputs.drop(categorical_cols, axis=1)

#     input_cols = numeric_cols + encoded_cols.tolist()

#     return {
#         'train_X': X_train,
#         'train_y': train_targets,
#         'val_X': X_val,
#         'val_y': val_targets,
#         'input_cols': input_cols,
#         'scaler': scaler,
#         'encoder': enc
#     }
