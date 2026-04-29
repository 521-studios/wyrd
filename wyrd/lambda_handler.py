"""AWS Lambda entry point. Wraps the Flask app via apig-wsgi."""

from apig_wsgi import make_lambda_handler

from wyrd.app import create_app

handler = make_lambda_handler(create_app(), binary_support=True)
