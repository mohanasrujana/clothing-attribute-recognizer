import argparse
from flask_ml.flask_ml_cli import MLCli
from .server import server

def main():
    """
    Main entry point for the Clothing Recognition CLI.

    This function sets up an argument parser, initializes the Flask-ML CLI
    with the server instance, and runs the CLI interface.
    """
    parser = argparse.ArgumentParser(description="Clothing Recognition CLI")
    cli = MLCli(server, parser)
    cli.run_cli()

if __name__ == "__main__":
    main()