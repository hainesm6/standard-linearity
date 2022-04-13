#!/usr/bin/env python
"""Tests for `standard_linearity` package. Run from repo directory."""

import pytest
from click.testing import CliRunner

import standard_linearity as sl
from standard_linearity import cli


@pytest.fixture
def example_data():
    from pathlib import Path

    path_to_csv = Path.cwd() / "csv_xlsx_files" / "22-03-25_19-01-53_pierce_gfp_standard.CSV"
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


def test_export_data(example_data, ols_fit):
    import shutil
    from pathlib import Path

    dir_path = Path.cwd() / "test_export_figs"
    Path.mkdir(dir_path)
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
    breakpoint()  # Need to optimise format of subplots in graphs.svg.
    shutil.rmtree(dir_path)


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'standard-linearity' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
