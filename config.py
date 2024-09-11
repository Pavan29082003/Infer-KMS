import os

class Config:
    GOOGLE_API_KEY = "AIzaSyDPCCwRJyLVLzv4QP7jwu8M9aEC87WrNMQ"
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    DEBUG = True
    IP =  os.getenv('IP_TEST', '')
    ENV = 'testing'

class DevelopmentConfig(Config):
    DEBUG = False
    IP = "13.232.28.221"
    ENV = 'development'

class ProductionConfig(Config):
    """Production config."""
    DEBUG = False
    ENV = 'production'

