from app import app

import configparser

if __name__ == "__main__":
    # load the window size config
    config = configparser.RawConfigParser()
    config.read('config.txt')
    app.config['WINDOW_SIZE'] = int(config.get('main', 'production_window_size'))

    app.run(debug=True) # TODO move this setting to .env
