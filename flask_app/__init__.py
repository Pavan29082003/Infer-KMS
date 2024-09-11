from flask import Flask
import os
import config as config


def create_app():
    app = Flask(__name__)

    from .search.routes import search as search_blueprint
    
    # Register blueprints
    app.register_blueprint(search_blueprint)

    return app

