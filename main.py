from controller import home, process, update_song
from flask import Flask

# Initialize the Flask application
app = Flask(__name__)

# Define URL routes and corresnponding funcions
app.add_url_rule('/', 'home', home)
app.add_url_rule('/update-song', 'update-song', update_song, methods=['POST'])
app.add_url_rule('/process', 'process', process, methods=['POST'])

if __name__ == '__main__':
    """
    Entry point of the Flask application.

    Starts the Flask development server for the Music Recommendation System.
    """
    app.run()
