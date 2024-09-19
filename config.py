import os

class Config:
    GOOGLE_API_KEY = "AIzaSyDPCCwRJyLVLzv4QP7jwu8M9aEC87WrNMQ"
    DEBUG = False
    TESTING = False
    SECRET_KEY = "d47e2a60-ff60-4b39-9961-72c8b63a9bc9" 
    IP = "13.127.207.184"

class TestingConfig(Config):
    GOOGLE_API_KEY = "AIzaSyDPCCwRJyLVLzv4QP7jwu8M9aEC87WrNMQ"
    DEBUG = False
    TESTING = False
    SECRET_KEY = "d47e2a60-ff60-4b39-9961-72c8b63a9bc9"     
    DEBUG = True
    IP =  "13.127.207.184"
    ENV = 'testing'

class DevelopmentConfig(Config):
    GOOGLE_API_KEY = "AIzaSyDPCCwRJyLVLzv4QP7jwu8M9aEC87WrNMQ"
    DEBUG = False
    TESTING = False
    SECRET_KEY = "d47e2a60-ff60-4b39-9961-72c8b63a9bc9"     
    DEBUG = False
    IP = "13.232.28.221"
    ENV = 'development'

class ProductionConfig(Config):
    
    DEBUG = False
    ENV = 'production'