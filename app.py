from app import app

app.config.from_pyfile('custom_configs.py')

if __name__ == "__main__":
    app.run(debug=True) # TODO move this setting to .env
