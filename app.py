from flask import Flask, render_template, request, Response
import subprocess
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video_path_global = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    global video_path_global

    file = request.files["video"]

    if file and file.filename.endswith(".mp4"):

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        video_path_global = filepath

    return "OK"


@app.route("/process")
def process():

    def generate():

        command = [
            "python",
            "-u",
            "predict.py",
            "--video_path", video_path_global,
            "--config", "config/size_invariant_timesformer.yaml",
            "--model_weights", "outputs/models/Model_checkpoint16",
            "--workers", "0"
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in iter(process.stdout.readline, ''):

            line = line.strip()

            if "Detecting faces" in line:
                yield "data: Detecting faces...\n\n"

            elif "Face detection completed" in line:
                yield "data: Face detection completed\n\n"

            elif "Cropping faces" in line:
                yield "data: Cropping faces from the video...\n\n"

            elif "Total faces extracted" in line:
                yield f"data: {line}\n\n"

            elif "Faces cropping completed" in line:
                yield "data: Faces cropping completed\n\n"

            elif "Clustering faces" in line:
                yield "data: Clustering faces...\n\n"

            elif "Faces clustering completed" in line:
                yield "data: Faces clustering completed\n\n"

            elif "Searching for fakes" in line:
                yield "data: Searching for fakes in the video...\n\n"

            elif "video is fake" in line.lower():
                yield "data: RESULT:FAKE\n\n"

            elif "video is real" in line.lower():
                yield "data: RESULT:REAL\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)