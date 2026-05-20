async function analyzeSentiment() {

    const text = document.getElementById("text").value;

    const response = await fetch(
        "/predict",
        {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text
            })
        }
    );

    const data = await response.json();

    document.getElementById("result").innerHTML = `

        <p>
            Sentiment:
            <b>${data.sentiment}</b>
        </p>

        <p>
            Confidence:
            <b>${data.confidence}</b>
        </p>
    `;
}