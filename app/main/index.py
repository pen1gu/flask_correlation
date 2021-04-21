from flask import Flask, render_template,redirect

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html',name = name)

app.run(host='0.0.0.0', port=5000, debug=True)
