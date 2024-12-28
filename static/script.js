document.getElementById('songs').addEventListener('change', async (e) =>{
    const selectedSong = e.target.value;
    const loading = document.getElementById('loading');
    const result = document.getElementById('audio');

    loading.style.display = 'block';
    result.innerHTML = ''; // Clear the current content

    try {
        const response = await fetch('/update-song', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ selectedSong })
        });

        const url = await response.json();

        if (url == null){
            loading.style.display = "none";
            result.innerHTML = `<div class="alert alert-danger">Track not available.</div>`
        }else{
            loading.style.display = 'none';
            result.innerHTML = `
                <audio controls class="w-100"><source src="${url}" type="audio/mpeg"></audio>
            `;
        }

    } catch (error) {
        loading.style.display = "none";
        result.innerHTML = `<div class="alert alert-danger">Track not available.</div>`
    }

});

document.getElementById('form').addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent form from refreshing the page

    const song = document.getElementById('songs').value;
    const algorithm = document.getElementById('clustering').value;
    const loading = document.getElementById('loading_two');
    const resultDiv = document.getElementById('result');

    loading.style.display = 'block';
    resultDiv.innerHTML = '';

    // Send POST request with selected option
    const response = await fetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song, algorithm })
    });

    // Parse the JSON response
    const data = await response.json();

    // Update the result section dynamically
    loading.style.display = "none";
    for (const key in data) {
        if (data[key].track_url !== null){
            resultDiv.innerHTML += `
                <p>${data[key].artist_name} - ${data[key].track_title}</p>
                <audio controls><source src="${data[key].track_url}" type="audio/mpeg"></audio>
            `;
        } else {
            resultDiv.innerHTML += `
                <p>${data[key].artist_name} - ${data[key].track_title}</p>
                <div class="alert alert-danger">Track not available.</div>
            `;
        }
    }
});