"""
Xplore DS :: Dataframe Tools Package
"""

@staticmethod
def print_dataframe_describe(self, dataframe):

    description = dataframe.describe()

    for row in description.iterrows():
        logging.info(row[0])
        for id in row[1].index:
            logging.info("Var: " + id + " {a:.3f}".format(a=(row[1][id])))

    return True


 @staticmethod
    def concatenate_pandas_columns(
        dataframe: pd = None, columns: list = None, conc_str: str = " "
    ) -> pd:
        for i in range(len(columns)):
            if i > 0:
                df = dataframe[columns[i - 1]] + conc_str + dataframe[columns[i]]

        return df

    @staticmethod
    def count_lists_pandas(dataframe: pd = None, column: str = None):
        count_list = []
        for index, row in dataframe.iterrows():
            count = len(row[column])
            count_list.append(count)
        column_name = column + "_list_count"
        dataframe[column_name] = count_list

        return dataframe, column_name

@staticmethod
    def get_list_from_pandas_list_rows(
        dataframe: pd = None, column: str = None
    ) -> list:
        tks_list = []
        for index, row in dataframe.iterrows():
            for item in row[column]:
                tks_list.append(item)
        return tks_list

    @staticmethod
    def save_dataframe(
        data: pd = None, folder_path: str = None, prefix: str = None
    ) -> bool:

        file_path = folder_path + prefix + ".tsv"
        data.to_csv(file_path, index=True, sep="\t", encoding="utf-8")
        logging.info("File saved in: " + file_path)

        return file_path