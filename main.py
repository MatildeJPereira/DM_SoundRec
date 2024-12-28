import intro
from controller import home, process

from flask import Flask

app = Flask(__name__)
app.add_url_rule('/', 'home', home)
app.add_url_rule('/process', 'process', process, methods=['POST'])


if __name__ == '__main__':
    app.run()
