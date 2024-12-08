import intro
from controller import home, cluster

from flask import Flask

app = Flask(__name__)
app.add_url_rule('/', 'home', home)
app.add_url_rule('/cluster', 'cluster', cluster)

if __name__ == '__main__':
    app.run()
