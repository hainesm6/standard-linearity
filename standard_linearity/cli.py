"""Console script for standard_linearity."""

import json
from pathlib import Path

import click
import statsmodels.api as sm
from patsy import dmatrices

from standard_linearity import export_data, import_data

DEFAULT_PLOT_CONFIG = {
    "calibration_attrs": None,
    "residual_attrs": None,
    "errors_attrs": None,
    "global_attrs": None,
    "figure_attrs": None,
}


@click.command()
@click.option(
    "--fig-format", type=str, default="svg", help="Format of the figure file. Refer to matplotlib.pyplot.savefig docs."
)
@click.option(
    "--fitting",
    "-f",
    type=click.Choice(["OLS", "WLS"], case_sensitive=False),
    default="OLS",
    help="""Algorithm to use for fitting standard curve.
    If WLS is used, (1/standard_concentrations)**2 are used as the weighting.""",
)
@click.option(
    "--header", "-h", type=int, default=0, help="Parameter used when reading data from csv. Refer to pd.read_csv()."
)
@click.option(
    "--nrows", "-n", type=int, default=None, help="Parameter used when reading data from csv. Refer to pd.read_csv()."
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False),
    default=None,
    help="Path to output directory. If not invoked, output directory is generated with a timestamp.",
)
@click.option(
    "--plot-config-file",
    "-p",
    type=click.File(),
    default=None,
    help="JSON file detailing configuration for plots.",
)
@click.option("--response-colname", "-r", required=True, help="Name of column with response data.")
@click.option(
    "--skip-rows", type=int, default=None, help="Parameter used when reading data from csv. Refer to pd.read_csv()."
)
@click.option(
    "--standards-colname",
    "-s",
    required=True,
    help="standards_colname: Name of column with standard concentrations in data.",
)
@click.argument(
    "input",
    required=True,
    type=click.Path(exists=True),
)
def main(
    fig_format,
    fitting,
    header,
    nrows,
    output_dir,
    plot_config_file,
    response_colname,
    standards_colname,
    skip_rows,
    input,
):
    """Main command for the standard-linearity CLI. # noqa: D417,D415

    Args:
        input: path to csv file - refer to pd.read_csv docs.

    """
    if output_dir:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True)
        except FileExistsError:
            pass
    data = import_data(
        path_to_csv=input,
        response_colname=response_colname,
        standards_colname=standards_colname,
        header=header,
        nrows=nrows,
        skip_rows=skip_rows,
    )
    response, predictors = dmatrices("standard_concentrations ~ response", data=data, return_type="dataframe")
    if fitting == "OLS":
        fitted_lm = sm.OLS(response, predictors).fit()
        student_resid = fitted_lm.outlier_test()["student_resid"]
    else:
        lm = sm.WLS(response, predictors, weights=1 / data["standard_concentrations"] ** 2)
        fitted_lm = lm.fit()
        pseudo_fit = sm.OLS(lm.wendog, lm.wexog).fit()
        student_resid = pseudo_fit.get_influence().resid_studentized
    if plot_config_file:
        plot_config = json.load(plot_config_file)
    else:
        plot_config = DEFAULT_PLOT_CONFIG.copy()
    export_data(
        data=data,
        student_resid=student_resid,
        fitted_lm=fitted_lm,
        dir_path=output_dir,
        fig_format=fig_format,
        calibration_attrs=plot_config["calibration_attrs"],
        residual_attrs=plot_config["residual_attrs"],
        errors_attrs=plot_config["errors_attrs"],
        global_attrs=plot_config["global_attrs"],
        figure_attrs=plot_config["figure_attrs"],
    )
