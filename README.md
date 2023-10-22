# AI-Powered Financial Footnotes Predictor

This project, developed during the Technica 2023 hackathon, tackles the Bloomberg Industry Group's challenge of creating an AI-centric solution to parse, display, and extract value from a corpus of text. Specifically, we focus on summarizing footnotes of SEC 10-K filings to predict next year’s footnote for Apple based on their historical fiscal data.

## Features
- Natural Language Understanding (NLU) to interpret financial data.
- A web-based interface for inputting queries and viewing predictions.
- Utilizes Vector DB for content retrieval and GPT-4 for generating predictions.

## Getting Started
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the Flask app using `python scripts/app.py`.

## Directory Structure
- `static/`: Contains CSS and font files.
- `scripts/`: Contains the Flask application script.
- `index.html`: The main HTML file for the web interface.
- `requirements.txt`: Lists the Python dependencies.

## Technologies Used
- Flask for the web application framework.
- GPT-4 for generating predictions.
- Vector DB for content retrieval.

## Acknowledgements
This project was inspired by the [Bloomberg Industry Group challenge](https://technica-2023.devpost.com/) at Technica 2023. The challenge emphasized creating AI-powered solutions to derive value from text data.

## License
This project is open source, under the MIT License.

For more information on the challenge and the hackathon, refer to the [Technica 2023 DevPost](https://technica-2023.devpost.com/).
