import intro
from controller import home

from flask import Flask

app = Flask(__name__)
app.add_url_rule('/', 'home', home)

if __name__ == '__main__':
    app.run()
