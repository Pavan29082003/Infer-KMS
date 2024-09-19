import config as config
from flask_app import create_app
import config as config
import os
from flask_cors import CORS
from flask_session import Session
from datetime import timedelta

config_class = os.getenv('FLASK_ENV', 'TestingConfig')
develop = config.DevelopmentConfig()
config_class =getattr(config, config_class)
ip = config_class.IP
gemini_key = config_class.GOOGLE_API_KEY
os.environ['GEMINI_API_KEY'] = gemini_key
os.environ['IP'] = ip

app = create_app()


app.secret_key=app.config['SECRET_KEY']

CORS(app)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=60)
Session(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80)