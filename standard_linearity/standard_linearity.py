"""Main module."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

# from patsy import dmatrices


def import_data(
    path_to_csv: str,
    response_colname: str,
    standards_colname: str,
    header: int = 0,
    nrows: int = None,
    skip_rows: int = None,
) -> pd.DataFrame:
    """Import standard curve data from a csv file.

    Args:
        path_to_csv: Refer to pd.read_csv docs.
        response_colname: Name of column with response data.
        standards_colname: Name of column with standard concentrations.
        header: Refer to pd.read_csv().
        nrows: Refer to pd.read_csv().
        skip_rows: Skips the first n rows when reading data.
        # kwargs: Additional arguments to parse to pd.read_csv().

    Returns:
        Formatted data as a dataframe.

    Raises:
        ValueError: If response_colname or standards_colname not in data.columns

    """
    data = pd.read_csv(path_to_csv, header=header, nrows=nrows)
    if skip_rows:
        data = data.iloc[skip_rows:, :]
    data.dropna(axis=1, how="all", inplace=True)
    data.dropna(inplace=True)
    data.rename({response_colname: "response", standards_colname: "standard_concentrations"}, axis=1, inplace=True)
    try:
        return data.loc[:, ["standard_concentrations", "response"]]
    except KeyError:
        raise ValueError("Check `response_colname` and `standards_colname` values are valid column names.")


def calculate_relative_errors(
    abs_val: npt.ArrayLike,
    fitted_lm,
) -> npt.ArrayLike:
    """Calculate relative errors from a linear model.

    Args:
        abs_val: Absolute value from data.
        fitted_lm: Linear model fitted to data. Refer to Statsmodel for examples and further docs.

    """
    return np.absolute((fitted_lm.resid / abs_val * 100).to_numpy())


def export_data(
    data: pd.DataFrame,
    student_resid: npt.ArrayLike,
    fitted_lm,
    dir_path: str = None,
    fig_format: str = "svg",
    calibration_attrs: dict = None,
    residual_attrs: dict = None,
    errors_attrs: dict = None,
    global_attrs: dict = None,
    figure_attrs: dict = None,
) -> None:
    """Export data from standard curve fitting to a directory.

    Export the following files to a directory:
         graphs.<fig_format> file - subplots of calibration curve, studentised residuals and relative errors.
         model_summary.txt - summary of the fitted model.
         linearity_parameters.txt - key parameters relating to the linearity of the standard curve.

    Args:
        dir_path: Path to output directory. If none,
            output directory is generated within current directory with timestamp.
        data: Imported data - refer to sl.import_data.
        student_resid: Studentised residuals.
        fitted_lm: Linear model fitted to data. Refer to Statsmodel for examples and further docs.
        fig_format: Format of the figure file. Refer to matplotlib.pyplot.savefig docs.
        calibration_attrs: Attributes to apply to the calibration curve axes.
        residual_attrs: Attributes to apply to the studentised residuals plot axes.
        errors_attrs: Attributes to apply to the % relative errors plot axes.
        global_attrs: Attributes to apply to all axes.
        figure_attrs: Attributes to apply to the figure.

    """
    if dir_path is None:
        time_stamp = datetime.now()
        dir_path = Path.cwd() / f"{time_stamp.strftime('%Y-%m-%d_%H%M%S')}_analysis"
        Path.mkdir(dir_path)
    else:
        dir_path = Path(dir_path)
    relative_errors = calculate_relative_errors(data["standard_concentrations"], fitted_lm)
    # Make plot figure
    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=[29.7 / 2.54, 21 / (2 * 2.54)])
    for ax in axes:
        ax.set_xlabel("concentration")
    axes[0].scatter(data["standard_concentrations"], data["response"], alpha=0.7, edgecolors="k")
    x_vals = np.linspace(0, int(round(data["standard_concentrations"].max())), int(1e3))
    axes[0].plot(x_vals, (x_vals - fitted_lm.params[0]) / fitted_lm.params[1], color="k")
    axes[0].set_ylabel("response")
    if calibration_attrs:
        for key, value in calibration_attrs.items():
            getattr(axes[0], key)(value)
    axes[1].scatter(data["standard_concentrations"], student_resid, alpha=0.7, edgecolors="k")
    axes[1].set_ylabel("studentised residuals")
    axes[1].axhline(y=2, color="r", linestyle="--")
    axes[1].axhline(y=-2, color="r", linestyle="--")
    if residual_attrs:
        for key, value in residual_attrs.items():
            getattr(axes[1], key)(value)
    axes[2].scatter(data["standard_concentrations"], relative_errors, alpha=0.7, edgecolors="k")
    axes[2].axhline(y=15, color='y', linestyle='--')
    axes[2].axhline(y=20, color='r', linestyle='--')
    axes[2].set_ylabel("% relative error")
    if errors_attrs:
        for key, value in errors_attrs.items():
            getattr(axes[2], key)(value)
    if global_attrs:
        for key, value in global_attrs.items():
            for ax in axes:
                getattr(ax, key)(value)
    figure.set_tight_layout(True)
    figure.savefig(dir_path / f"graphs.{fig_format}")
    if figure_attrs:
        for key, value in figure_attrs.items():
            getattr(figure, key)(value)
    # Save parameters to txt files
    with open(dir_path / "model_summary.txt", "w") as file:
        file.write(fitted_lm.summary().as_text())
    with open(dir_path / "linearity_parameters.txt", "w") as file:
        file.write(f"{'Max absolute studentised residual:': <35} {max(np.absolute(student_resid))}\n")
        file.write(f"{'Max % relative error:': <35} {max(relative_errors)}\n")
        file.write(f"{'Min concentration (μg/mL):': <35} {np.min(data['standard_concentrations'])}\n")
        file.write(f"{'Max concentration (μg/mL):': <35} {np.max(data['standard_concentrations'])}\n")
