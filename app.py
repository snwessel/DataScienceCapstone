from app import app

app.config.from_pyfile('config.py')

if __name__ == "__main__":
    app.run(debug=True) # TODO move this setting to .env
