from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import glob
import os
import time

app = Flask(__name__)
CORS(app)

@app.route('/live')
def live_data():
    try:
        files = glob.glob("stroke_logs/*.csv")

        if not files:
            return jsonify({"error": "No CSV found"})

        latest_file = max(files, key=os.path.getmtime)

        df = pd.read_csv(latest_file)

        if df.empty:
            return jsonify({"error": "CSV empty"})

        latest = df.iloc[-1]

        return jsonify({
            "risk": float(latest.get("risk", 0)),
            "mouth": float(latest.get("mouth", 0)),
            "eye": float(latest.get("eye", 0)),
            "tilt": float(latest.get("tilt", 0)),
            "arm": float(latest.get("arm", 0)),
            "fps": float(latest.get("fps", 0)),
            "alert": int(latest.get("alert_active", 0))
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000)