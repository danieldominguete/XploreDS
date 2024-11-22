"""
Xplore DS :: Exploratory Data Analysis
"""

import sys, os
from pathlib import Path
import pandas as pd

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xploreds.data_handler.file import create_folder
from xploreds.data_visualization.data_viz_plotly import (
    plot_bar,
    plot_lines,
    plot_boxplot,
)
from xploreds.data_analysis.statistics import get_information_value


def descriptive_analysis(
    data: pd,
    numerical_variables: list = None,
    categorical_variables: list = None,
    view_plots: bool = False,
    save_plots: bool = False,
    save_analysis: bool = False,
    output_folder_path: str = None,
    prefix_label: str = None,
    log: object = None,
) -> None:

    # ----------------------------------------------------------
    # numerical variable analysis
    if len(numerical_variables) > 0:
        if log:
            log.subtitle("Numerical variables infos")

        num_variables_analysis = pd.DataFrame(
            index=numerical_variables, columns=["not_nulls", "not_nulls_perc", "mean"]
        )

        for var in numerical_variables:

            # not nulls
            value = data[var].notna().sum()
            log.info("Not Null Var: " + str(var) + " = {:.4f}".format(value))
            num_variables_analysis["not_nulls"].loc[var] = value

            # not nulls (%)
            value = 100 * (data[var].notna().sum() / data[var].shape[0])
            log.info("Not Null Var: " + str(var) + " = {:.2f}%".format(value))
            num_variables_analysis["not_nulls_perc"].loc[var] = value

            # average
            value = data[var].mean()
            log.info("Mean Var: " + str(var) + " = {:.4f}".format(value))
            num_variables_analysis["mean"].loc[var] = value

        num_variables_analysis = num_variables_analysis.reset_index(
            names=["variable_name"]
        )

    # ----------------------------------------------------------
    # categorical variables analysis
    if len(categorical_variables) > 0:
        if log:
            log.subtitle("Categorical variables infos")

        cat_variables_analysis = pd.DataFrame(
            index=categorical_variables,
            columns=["not_nulls", "not_nulls_perc", "unique"],
        )

        for var in categorical_variables:

            # not nulls
            value = data[var].notna().sum()
            log.info("Not Null Var: " + str(var) + " = {:.4f}".format(value))
            cat_variables_analysis["not_nulls"].loc[var] = value

            # not nulls (%)
            value = 100 * (data[var].notna().sum() / data[var].shape[0])
            log.info("Not Null Var: " + str(var) + " = {:.2f}%".format(value))
            cat_variables_analysis["not_nulls_perc"].loc[var] = value

            # unique
            value = len(data[var].unique())
            log.info("Unique Var: " + str(var) + " = {:.4f}".format(value))
            cat_variables_analysis["unique"].loc[var] = value

        cat_variables_analysis = cat_variables_analysis.reset_index(
            names=["variable_name"]
        )

    # ----------------------------------------------------------
    # saving statistics
    if save_analysis:

        if log:
            log.info("Saving descriptive analysis...")

        full_path = (
            output_folder_path + "/reports/" + prefix_label + "describe_statistics.xlsx"
        )

        # verificando se a pasta existe caso contrario criar a pasta
        create_folder(os.path.dirname(full_path))

        # Multiple DataFrames to different sheets
        with pd.ExcelWriter(full_path) as writer:
            num_variables_analysis.to_excel(
                writer, sheet_name="numerical_variables", index=False
            )
            cat_variables_analysis.to_excel(
                writer, sheet_name="categorical_variables", index=False
            )

        if log:
            log.info("Descriptive analysis saved in " + full_path)

    if view_plots or save_plots:

        if log:
            log.info("Plotting descriptive analysis...")

        for stat in num_variables_analysis.columns:
            if stat != "variable_name":
                plot_bar(
                    data=num_variables_analysis,
                    x_col_name=num_variables_analysis["variable_name"],
                    y_col_name=num_variables_analysis[stat],
                    title="Metric of " + stat,
                    view_chart=view_plots,
                    save_chart=save_plots,
                    file_path_image=output_folder_path
                    + "/charts/"
                    + prefix_label
                    + var
                    + ".png",
                )

        for stat in cat_variables_analysis.columns:
            if stat != "variable_name":
                plot_bar(
                    data=cat_variables_analysis,
                    x_col_name=cat_variables_analysis["variable_name"],
                    y_col_name=cat_variables_analysis[stat],
                    title="Metric of " + stat,
                    view_chart=view_plots,
                    save_chart=save_plots,
                    file_path_image=output_folder_path
                    + "/charts/"
                    + prefix_label
                    + var
                    + ".png",
                )


def trend_analysis(
    data: pd,
    date_col_name: str,
    date_col_format: str = "%Y-%m-%d",
    date_trunc_by: str = None,
    numerical_variables: list = None,
    categorical_variables: list = None,
    view_plots: bool = False,
    save_plots: bool = False,
    save_analysis: bool = False,
    output_folder_path: str = None,
    prefix_label: str = None,
    log: object = None,
) -> None:
    """
    Performs trend analysis on time series data, analyzing both numerical and categorical variables over time.

    Args:
        data (pd.DataFrame): Input DataFrame containing the time series data.
        date_col_name (str): Name of the column containing datetime values.
        date_col_format (str, optional): Format of the date string. Defaults to "%Y-%m-%d".
        date_trunc_by (str, optional): Time unit to truncate dates by ('S', 'T', 'H', 'D', 'W', 'M', 'Q', 'Y', etc.). Defaults to None.
        numerical_variables (list, optional): List of numerical column names to analyze. Defaults to None.
        categorical_variables (list, optional): List of categorical column names to analyze. Defaults to None.
        view_plots (bool, optional): If True, displays the generated plots. Defaults to False.
        save_plots (bool, optional): If True, saves the generated plots to files. Defaults to False.
        save_analysis (bool, optional): If True, saves analysis results to Excel file. Defaults to False.
        output_folder_path (str, optional): Path to save output files. Required if save_plots or save_analysis is True.
        prefix_label (str, optional): Prefix to add to output filenames. Defaults to None.
        log (object, optional): Logger object for output messages. Defaults to None.

    Returns:
        None
    """

    # ----------------------------------------------------------
    # datetime index
    data.index = pd.to_datetime(data[date_col_name], format=date_col_format)
    data = data.sort_index(ascending=True)

    # ----------------------------------------------------------
    # trunc by date
    if date_trunc_by is not None:
        data["_dt_trunc"] = data.index.to_period(date_trunc_by).to_timestamp()

    # ----------------------------------------------------------
    # numerical variable analysis
    if len(numerical_variables) > 0:
        if log:
            log.subtitle("Numerical variables trends")

        num_variables_analysis = pd.DataFrame(
            index=numerical_variables, columns=["not_nulls", "not_nulls_perc"]
        )

        for var in numerical_variables:

            # not nulls
            value = data[var].notna().sum()
            log.info("Not Null Var: " + str(var) + " = {:.4f}".format(value))
            num_variables_analysis["not_nulls"].loc[var] = value

            # not nulls (%)
            value = 100 * (data[var].notna().sum() / data[var].shape[0])
            log.info("Not Null Var: " + str(var) + " = {:.2f}%".format(value))
            num_variables_analysis["not_nulls_perc"].loc[var] = value

        num_variables_analysis = num_variables_analysis.reset_index(
            names=["variable_name"]
        )

    # ----------------------------------------------------------
    # categorical variables analysis
    if len(categorical_variables) > 0:
        if log:
            log.subtitle("Categorical variables infos")

        cat_variables_analysis = pd.DataFrame(
            index=categorical_variables,
            columns=["not_nulls", "not_nulls_perc", "unique"],
        )

        for var in categorical_variables:

            # not nulls
            value = data[var].notna().sum()
            log.info("Not Null Var: " + str(var) + " = {:.4f}".format(value))
            cat_variables_analysis["not_nulls"].loc[var] = value

            # not nulls (%)
            value = 100 * (data[var].notna().sum() / data[var].shape[0])
            log.info("Not Null Var: " + str(var) + " = {:.2f}%".format(value))
            cat_variables_analysis["not_nulls_perc"].loc[var] = value

        cat_variables_analysis = cat_variables_analysis.reset_index(
            names=["variable_name"]
        )

    # ----------------------------------------------------------
    # saving statistics
    if save_analysis:

        if log:
            log.info("Saving trend analysis...")

        full_path = (
            output_folder_path + "/reports/" + prefix_label + "trend_statistics.xlsx"
        )

        # verificando se a pasta existe caso contrario criar a pasta
        create_folder(os.path.dirname(full_path))

        # Multiple DataFrames to different sheets
        with pd.ExcelWriter(full_path) as writer:
            num_variables_analysis.to_excel(
                writer, sheet_name="numerical_variables", index=False
            )
            cat_variables_analysis.to_excel(
                writer, sheet_name="categorical_variables", index=False
            )

        if log:
            log.info("Trend analysis saved in " + full_path)

    # ----------------------------------------------------------
    # saving charts
    if view_plots or save_plots:

        if log:
            log.info("Plotting trend analysis...")

        for stat in num_variables_analysis.columns:
            if stat != "variable_name":
                plot_bar(
                    data=num_variables_analysis,
                    x_col_name=num_variables_analysis["variable_name"],
                    y_col_name=num_variables_analysis[stat],
                    title="Metric of " + stat,
                    view_chart=view_plots,
                    save_chart=save_plots,
                    file_path_image=output_folder_path
                    + "/charts/"
                    + prefix_label
                    + stat
                    + ".png",
                )

        for stat in cat_variables_analysis.columns:
            if stat != "variable_name":
                plot_bar(
                    data=cat_variables_analysis,
                    x_col_name=cat_variables_analysis["variable_name"],
                    y_col_name=cat_variables_analysis[stat],
                    title="Metric of " + stat,
                    view_chart=view_plots,
                    save_chart=save_plots,
                    file_path_image=output_folder_path
                    + "/charts/"
                    + prefix_label
                    + stat
                    + ".png",
                )

        for var in numerical_variables:

            plot_lines(
                data=data,
                x_col_name=data.index,
                y_col_name=data[var],
                title="Trend of " + var,
                view_chart=view_plots,
                save_chart=save_plots,
                file_path_image=output_folder_path
                + "/charts/"
                + prefix_label
                + "_trend_"
                + var
                + ".png",
            )

            if date_trunc_by is not None:

                plot_boxplot(
                    data=data,
                    x_col_name=data["_dt_trunc"],
                    y_col_name=data[var],
                    title="Distribution trend of " + var,
                    view_chart=view_plots,
                    save_chart=save_plots,
                    file_path_image=output_folder_path
                    + "/charts/"
                    + prefix_label
                    + "_dist_trend_"
                    + var
                    + ".png",
                )

        for var in categorical_variables:

            if date_trunc_by is not None:

                data_agg = (
                    data.groupby(["_dt_trunc", var])
                    .agg(
                        {
                            var: [
                                ("count", "count"),
                                ("unique", "nunique"),
                                ("nulls", lambda x: x.isnull().sum()),
                                ("not_nulls", lambda x: x.notnull().sum()),
                                # Calculate percentage within each _dt_trunc group
                                (
                                    "perc",
                                    lambda x: 100
                                    * len(x)
                                    / len(x.groupby(level=0).transform("count")),
                                ),
                            ]
                        }
                    )
                    .reset_index()
                )

                data_agg.columns = (
                    ["_dt_trunc"]
                    + [var]
                    + [f"{col[1]}" for col in data_agg.columns[2:]]
                )

                plot_bar(
                    data=data_agg,
                    x_col_name=data_agg["_dt_trunc"],
                    y_col_name=data_agg["count"],
                    color_col_name=data_agg[var],
                    title="Distribution trend of " + var,
                    view_chart=view_plots,
                    save_chart=save_plots,
                    file_path_image=output_folder_path
                    + "/charts/"
                    + prefix_label
                    + "_dist_trend_"
                    + var
                    + ".png",
                )

                # Calculate counts and overall percentage
                data_agg = (
                    data.groupby(["_dt_trunc", var]).size().reset_index(name="count")
                )

                # Calculate percentages within each _dt_trunc group (relative to time period)
                data_agg["perc_by_period"] = data_agg.groupby("_dt_trunc")[
                    "count"
                ].transform(lambda x: 100 * x / x.sum())

                # Calculate overall percentage (relative to total dataset)
                total_records = data_agg["count"].sum()
                data_agg["perc_total"] = 100 * data_agg["count"] / total_records

                # Plot with overall percentages
                plot_bar(
                    data=data_agg,
                    x_col_name="_dt_trunc",
                    y_col_name="perc_by_period",  # Using overall percentage
                    color_col_name=var,
                    title=f"Distribution trend of {var} (% of total)",
                    view_chart=view_plots,
                    save_chart=save_plots,
                    file_path_image=output_folder_path
                    + "/charts/"
                    + prefix_label
                    + "_dist_trend_percent_total_"
                    + var
                    + ".png",
                )


def categorical_target_association_analysis(
    data: pd,
    date_col_name: str,
    date_col_format: str = "%Y-%m-%d",
    date_trunc_by: str = None,
    numerical_variables: list = None,
    categorical_variables: list = None,
    target_col_name: str = None,
    view_plots: bool = False,
    save_plots: bool = False,
    save_analysis: bool = False,
    output_folder_path: str = None,
    prefix_label: str = None,
    log: object = None,
) -> None:

    # information value
    df_iv, df_woe = get_information_value(
        data=data,
        y_true_numeric_column_name=target_col_name,
        var_categorical_column_name=categorical_variables,
        var_numeric_column_names=numerical_variables,
        log=log,
    )

    if view_plots or save_plots:

        if log:
            log.info("Plotting categorical target association analysis...")

        plot_bar(
            data=df_iv,
            y_col_name=df_iv["variable"],
            x_col_name=df_iv["iv"],
            text_col_name=df_iv["analysis"],
            title="Information Value with " + target_col_name,
            orientation="h",
            view_chart=view_plots,
            save_chart=save_plots,
            file_path_image=output_folder_path
            + "/charts/"
            + prefix_label
            + "iv_"
            + ".png",
        )

        plot_bar(
            data=df_woe,
            y_col_name=df_woe["variable"],
            x_col_name=df_woe["woe"].abs(),
            color_col_name=df_woe["variable_group"],
            barmode="stack",
            orientation="h",
            title="WoE with " + target_col_name,
            view_chart=view_plots,
            save_chart=save_plots,
            file_path_image=output_folder_path
            + "/charts/"
            + prefix_label
            + "woe_"
            + ".png",
        )

    # saving statistics
    if save_analysis:

        if log:
            log.info("Saving categorical target association analysis...")

        full_path = (
            output_folder_path + "/reports/" + prefix_label + "association_target.xlsx"
        )

        # verificando se a pasta existe caso contrario criar a pasta
        create_folder(os.path.dirname(full_path))

        # Multiple DataFrames to different sheets
        with pd.ExcelWriter(full_path) as writer:
            df_iv.to_excel(writer, sheet_name="information_value", index=False)
            df_woe.to_excel(writer, sheet_name="woe", index=False)

        if log:
            log.info("Categorical target association analysis saved in " + full_path)
