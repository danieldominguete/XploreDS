"""
Xplore DS :: Encoding Categorical Variables
"""

import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
)

from pathlib import Path
import sys

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xploreds.data_schemas.dataset_config import EncodingMethod


def encoder_variable_fit(
    data: pd,
    variable_column_name: str,
    encode_method: EncodingMethod,
    log: object = None,
):
    """
    Fits an encoder on the specified column data.

    Args:
        data: Input DataFrame
        variable_column_name: Column to encode
        encode_method: Encoding method to use
        log: Optional logger instance

    Returns:
        Fitted encoder instance
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    if variable_column_name not in data.columns:
        raise ValueError(f"Column {variable_column_name} not found in DataFrame")

    if encode_method == EncodingMethod.one_hot_encoder:

        encoder = OneHotEncoder(
            categories="auto",
            drop=None,
            dtype=int,
            handle_unknown="ignore",
        )

    else:
        encoder = None
        return encoder

    try:
        encoder.fit(data[variable_column_name].values.reshape(-1, 1))
        return encoder
    except Exception as e:
        if log:
            log.error(f"Error fitting encoder: {str(e)}")
        raise


def encoder_variable_transform(
    data: pd,
    variable_column_name: str,
    encoder,
    log: object = None,
):
    """
    Transforms data using the fitted encoder.

    Args:
        data: Input DataFrame
        variable_column_name: Column to encode
        encoder: Fitted encoder instance
        log: Optional logger instance

    Returns:
        Tuple containing:
        - Transformed DataFrame
        - List of encoded variable names
    """

    encoded_variables = []
    try:
        if encoder:
            transf = encoder.transform(data[variable_column_name].values.reshape(-1, 1))
            ohe_df = pd.DataFrame(transf.toarray())

            encoded_variables = []
            feature_names_processed = encoder.get_feature_names_out(
                [variable_column_name]
            )

            for item in ohe_df.columns.to_list():
                encoded_variables.append(feature_names_processed[item])
            ohe_df.columns = encoded_variables

            data = pd.concat([data, ohe_df], axis=1)

        else:
            encoded_variables.append(variable_column_name)

        return data, encoded_variables
    except Exception as e:
        if log:
            log.error(f"Error transforming data: {str(e)}")
        raise


def encoder_variable_fit_transform(
    data: pd,
    variable_column_name: str,
    encode_method: EncodingMethod,
    log: object,
):
    """
    Combines fitting and transformation in one step.

    Args:
        data: Input DataFrame
        variable_column_name: Column to encode
        encode_method: Encoding method to use
        log: Optional logger instance

    Returns:
        Tuple containing:
        - Transformed DataFrame
        - List of encoded variable names
    """

    if log:
        log.info(f"Encoding variable {variable_column_name} with {encode_method}...")

    encoder = encoder_variable_fit(
        data=data,
        variable_column_name=variable_column_name,
        encode_method=encode_method,
        log=log,
    )

    data, encoded_variable = encoder_variable_transform(
        data=data,
        variable_column_name=variable_column_name,
        encoder=encoder,
        log=log,
    )

    return data, encoded_variable


# def label_2_one_hot_fit_transform(data: pd, columns: list) -> pd:

#     var_list = []
#     val_working = []
#     encoders_int = []
#     encoders_hot = []
#     encoders_bin = []
#     int_to_cat_dict_list = []
#     cat_to_int_dict_list = []

#     for var in columns:

#         # variable reference
#         var_list.append(var)

#         # Encoding with one hot / dummy vector
#         encoder_hot = OneHotEncoder(
#             categories="auto",
#             drop=None,
#             sparse=False,
#             dtype=np.int,
#             handle_unknown="ignore",
#         )

#         transf_hot = encoder_hot.fit_transform(data[var].to_numpy().reshape(-1, 1))

#         col_temp = []
#         for item in encoder_hot.get_feature_names():
#             val_working.append(var + "_" + item)
#             col_temp.append(var + "_" + item)

#         ohe_df = pd.DataFrame(transf_hot, columns=col_temp)
#         data = pd.concat([data, ohe_df], axis=1)
#         encoders_hot.append(encoder_hot)

#         # Creating dictionaries convertion
#         categories = encoder_hot.categories_[0]
#         int_to_cat = {i: categories[i] for i in range(0, len(categories))}
#         cat_to_int = {categories[i]: i for i in range(0, len(categories))}

#         int_to_cat_dict_list.append(int_to_cat)
#         cat_to_int_dict_list.append(cat_to_int)

#     return (
#         data,
#         val_working,
#         var_list,
#         encoders_int,
#         encoders_hot,
#         encoders_bin,
#         int_to_cat_dict_list,
#         cat_to_int_dict_list,
#     )


# # def label_2_one_hot_fit(data: pd, columns: list) -> pd:

# #     var_list = []
# #     val_working = []
# #     encoders_int = []
# #     encoders_hot = []
# #     encoders_bin = []
# #     int_to_cat_dict_list = []
# #     cat_to_int_dict_list = []

# #     for var in columns:

# #         if type == "int":

# #             # variable reference
# #             var_list.append(var)

# #             # Encoding with integer identification
# #             col_int = []
# #             encoder_int = OrdinalEncoder(categories="auto", dtype=np.int)
# #             transf_int = encoder_int.fit_transform(data[var].to_numpy().reshape(-1, 1))
# #             col_int.append(var + "_" + "int")
# #             int_df = pd.DataFrame(transf_int, columns=col_int)
# #             data = pd.concat([data, int_df], axis=1)
# #             encoders_int.append(encoder_int)
# #             val_working.append(var + "_" + "int")

# #             # Creating dictionaries convertion
# #             categories = encoder_int.categories_[0]
# #             int_to_cat = {i: categories[i] for i in range(0, len(categories))}
# #             cat_to_int = {categories[i]: i for i in range(0, len(categories))}

# #             int_to_cat_dict_list.append(int_to_cat)
# #             cat_to_int_dict_list.append(cat_to_int)

# #         elif type == "one_hot":

# #             # variable reference
# #             var_list.append(var)

# #             # Encoding with one hot / dummy vector
# #             encoder_hot = OneHotEncoder(
# #                 categories="auto",
# #                 drop=None,
# #                 sparse=False,
# #                 dtype=np.int,
# #                 handle_unknown="ignore",
# #             )
# #             transf_hot = encoder_hot.fit_transform(data[var].to_numpy().reshape(-1, 1))

# #             col_temp = []
# #             for item in encoder_hot.get_feature_names():
# #                 val_working.append(var + "_" + item)
# #                 col_temp.append(var + "_" + item)

# #             ohe_df = pd.DataFrame(transf_hot, columns=col_temp)
# #             data = pd.concat([data, ohe_df], axis=1)
# #             encoders_hot.append(encoder_hot)

# #             # Creating dictionaries convertion
# #             categories = encoder_hot.categories_[0]
# #             int_to_cat = {i: categories[i] for i in range(0, len(categories))}
# #             cat_to_int = {categories[i]: i for i in range(0, len(categories))}

# #             int_to_cat_dict_list.append(int_to_cat)
# #             cat_to_int_dict_list.append(cat_to_int)

# #         elif type == "binarizer":

# #             # variable reference
# #             var_list.append(var)

# #             # Encoding with integer identification
# #             col_bin = []
# #             encoder_bin = LabelBinarizer()
# #             transf_int = encoder_bin.fit_transform(data[var].to_numpy().reshape(-1, 1))
# #             col_bin.append(var + "_" + "bin")
# #             val_working.append(var + "_" + "bin")
# #             bin_df = pd.DataFrame(transf_int, columns=col_bin)
# #             data = pd.concat([data, bin_df], axis=1)
# #             encoders_bin.append(encoder_bin)

# #             # Decoding example
# #             # test = encoder_int.inverse_transform(bin_df)

# #             # Creating dictionaries convertion
# #             categories = encoder_bin.classes_
# #             int_to_cat = {i: categories[i] for i in range(0, len(categories))}
# #             cat_to_int = {categories[i]: i for i in range(0, len(categories))}

# #             int_to_cat_dict_list.append(int_to_cat)
# #             cat_to_int_dict_list.append(cat_to_int)

# #         else:
# #             logging.error("Categorical encoder not valid")

# #     return (
# #         data,
# #         val_working,
# #         var_list,
# #         encoders_int,
# #         encoders_hot,
# #         encoders_bin,
# #         int_to_cat_dict_list,
# #         cat_to_int_dict_list,
# #     )
