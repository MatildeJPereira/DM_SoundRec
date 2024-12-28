document.getElementById('form').addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent form from refreshing the page

    const song = document.getElementById('songs').value;
    const algorithm = document.getElementById('clustering').value;

    // Send POST request with selected option
    const response = await fetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song, algorithm })
    });

    // Parse the JSON response
    const data = await response.json();

    console.log(data)
    // Update the result section dynamically TODO change this to the title and audio
    data.forEach((song) => {
        console.log(song)
        document.getElementById('result').innerHTML += `
            <audio controls><source src="${song}" type="audio/mpeg"></audio>
        `;
    })
});