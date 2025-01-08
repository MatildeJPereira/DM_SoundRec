from controller import home, process, update_song

from flask import Flask

app = Flask(__name__)
app.add_url_rule('/', 'home', home)
app.add_url_rule('/update-song', 'update-song', update_song, methods=['POST'])
app.add_url_rule('/process', 'process', process, methods=['POST'])


if __name__ == '__main__':
    app.run()
