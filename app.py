import os
from flask import Flask
from flask_cors import CORS

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.api.api_db import bp_db
from src.api.api_init_weight import bp_init_weight
from src.api.api_rec import bp_rec
from src.api.api_train import bp_train

app = Flask(__name__)
CORS(app)

app.register_blueprint(bp_db)
app.register_blueprint(bp_init_weight)
app.register_blueprint(bp_rec)
app.register_blueprint(bp_train)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 5000), debug=True)