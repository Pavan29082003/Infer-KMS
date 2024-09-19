import config as config
from flask_app import create_app
import config as config
import os
from flask_cors import CORS
from flask_session import Session
from datetime import timedelta

config_class = os.getenv('FLASK_ENV', 'TestingConfig')
config_class =getattr(config, config_class)
os.environ['IP'] = config_class.IP
os.environ['GEMINI_API_KEY'] = config_class.GEMINI_API_KEY
app = create_app()
app.secret_key=app.config['SECRET_KEY']
CORS(app)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=60)
Session(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80)