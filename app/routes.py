from flask import request, jsonify
import yaml
from app import app
from tsforecaster import train_model


@app.route("/train", methods=["POST"])
def train():
    config_params = request.json
    # Validate and sanitize the received parameters

    # Save the configuration parameters as a YAML file
    with open("config.yaml", "w") as file:
        yaml.dump(config_params, file)

    try:
        # Train the model using the tsforecaster library
        train_model("config.yaml")
        return jsonify({"message": "Model training completed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
