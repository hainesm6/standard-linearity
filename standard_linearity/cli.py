"""Console script for standard_linearity."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("standard-linearity")
    click.echo("=" * len("standard-linearity"))
    click.echo("A small project to assess the linearity of standard calibration curves")


if __name__ == "__main__":
    main()  # pragma: no cover
