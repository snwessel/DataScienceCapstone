from app import app
import os

is_prod = os.environ.get('IS_HEROKU', None)

if __name__ == "__main__":
    if is_prod:
        app.run()
    else:
        app.run(debug=True)
