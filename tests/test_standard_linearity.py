#!/usr/bin/env python
"""Tests for `standard_linearity` package. Run from repo directory."""

import pytest
from click.testing import CliRunner

import standard_linearity as sl
from standard_linearity import cli


@pytest.fixture
def dir_path():
    from pathlib import Path

    return Path.cwd() / "test_export_data"


@pytest.fixture
def path_to_csv():
    from pathlib import Path

    return Path.cwd() / "csv_xlsx_files" / "22-03-25_19-01-53_pierce_gfp_standard.CSV"


@pytest.fixture
def example_data(path_to_csv):
    example_data = sl.import_data(
        path_to_csv=path_to_csv,
        standards_colname="Standard Concentrations",
        response_colname=" Blank corrected based on Raw Data (F: 482-16/525-20)",
        header=3,
    )
    return example_data


@pytest.fixture
def better_data(path_to_csv):
    example_data = sl.import_data(
        path_to_csv=path_to_csv,
        standards_colname="Standard Concentrations",
        response_colname=" Blank corrected based on Raw Data (F: 482-16/525-20)",
        header=3,
    )
    return example_data


@pytest.fixture
def ols_fit(example_data):
    from patsy import dmatrices
    from statsmodels.api import OLS

    response, predictors = dmatrices("standard_concentrations ~ response", data=example_data, return_type="dataframe")
    return OLS(response, predictors).fit()


def test_dataframe_import(example_data):
    assert len(example_data) == 39


def test_export_data(dir_path, example_data, ols_fit):
    import shutil
    from pathlib import Path

    try:
        Path.mkdir(dir_path)
    except FileExistsError:
        pass
    global_attrs = {
        "set_xlabel": "concentration (μg/mL)",
    }
    calibration_attrs = {
        "set_ylabel": "fluorescence (AU)",
    }
    residual_attrs = {"set_xscale": "log"}
    errors_attrs = {"set_xscale": "log"}
    student_resid = ols_fit.outlier_test()["student_resid"]
    sl.export_data(
        dir_path=dir_path,
        data=example_data,
        student_resid=student_resid,
        fitted_lm=ols_fit,
        global_attrs=global_attrs,
        calibration_attrs=calibration_attrs,
        residual_attrs=residual_attrs,
        errors_attrs=errors_attrs,
    )
    assert len([file for file in dir_path.iterdir()]) == 3
    shutil.rmtree(dir_path)


def test_cli_help():
    """Test the CLI."""
    runner = CliRunner()
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert 'Show this message and exit.' in help_result.output


def test_cli_ols(dir_path, path_to_csv):
    import shutil
    from pathlib import Path

    try:
        Path.mkdir(dir_path)
    except FileExistsError:
        pass

    try:
        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [
                "-o",
                f"{dir_path}",
                "-p",
                "./json_files/test-plot-config.json",
                "-h",
                3,
                "-r",
                " Blank corrected based on Raw Data (F: 482-16/525-20)",
                "-s",
                "Standard Concentrations",
                f"{path_to_csv}",
            ],
        )
        print(result.output)
        assert result.exit_code == 0
        assert len([file for file in dir_path.iterdir()]) == 3
    finally:
        shutil.rmtree(dir_path)


def test_cli_wls(dir_path, path_to_csv):
    import shutil
    from pathlib import Path

    try:
        Path.mkdir(dir_path)
    except FileExistsError:
        pass
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [
                "-f",
                "WLS",
                "-h",
                3,
                "-n",
                30,
                "-o",
                "./test_export_data",
                "-p",
                "./json_files/test-plot-config.json",
                "-r",
                " Blank corrected based on Raw Data (F: 482-16/525-20)",
                "--skip-rows",
                3,
                "-s",
                "Standard Concentrations",
                f"{path_to_csv}",
            ],
        )
        print(result.output)
        assert result.exit_code == 0
        assert len([file for file in dir_path.iterdir()]) == 3
    finally:
        shutil.rmtree(dir_path)
