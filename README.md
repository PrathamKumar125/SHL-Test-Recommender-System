# SHL Test Recommender System

This system recommends SHL tests based on job descriptions or natural language queries.

## Huggingface Space Deployment:
- **Frontend**: https://huggingface.co/spaces/pratham0011/SHL-Test-Recommender-Gradio
- **Backend**: https://huggingface.co/spaces/pratham0011/SHL-Test-Recommender-API

<br>

![Screenshot 2025-04-10 050728](https://github.com/user-attachments/assets/d5df89e1-35f6-4304-8eef-3d7999246f69)

<br>

![image](https://github.com/user-attachments/assets/64c6b48e-461c-4cc0-a2bd-c25df851ea2d)


<br>
## Project Structure

The project is divided into two main components:

- **Backend**: A FastAPI application that provides the recommendation API
- **Frontend**: A Gradio interface for interacting with the API

## Backend

The backend is a FastAPI application that provides endpoints for recommending SHL tests. It uses a sentence transformer model to encode job descriptions and find the most similar SHL tests.

### Model Selection

The backend can use one of two text generation models:

1. **Gemini API**: If the `GEMINI_API_KEY` environment variable is set, the API will use Google's Gemini API for text generation.
2. **Local Qwen Model**: If no Gemini API key is provided, the API will fall back to using a local Qwen model.

### API Quota Handling

The system is designed to handle Gemini API quota limitations gracefully:

- If the Gemini API quota is exceeded or rate limits are hit, the system will automatically fall back to the local Qwen model.
- Once the system switches to the local model due to API limitations, it will continue using the local model for the remainder of the session to avoid further API errors.
- The health endpoint will report which model is currently active.

### API Endpoints

- `/health`: Check the health of the API
- `/`: Welcome message
- `/recommend`: Get recommendations based on a query

## Frontend

The frontend is a Gradio interface that allows users to input job descriptions or URLs and get recommendations for SHL tests. It communicates with the backend API to get recommendations.

The frontend provides an intuitive interface for users to:

- Enter job descriptions or queries directly
- Provide URLs to job postings for analysis
- View recommended SHL tests with descriptions
- Control the number of recommendations displayed

## Running Locally

To run the system locally:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Optional: Set `GEMINI_API_KEY` environment variable to use the Gemini model
4. Start the backend:
```bash
cd backend
python main.py
```

5. Start the frontend:
```bash
python frontend.py
```

The backend will be available at http://localhost:8000 and the frontend at http://localhost:7860.

## Performance

The system uses semantic similarity to match job requirements with appropriate SHL tests, providing:

- Fast response times
- Accurate recommendations
- Detailed test descriptions

## License

This project is licensed under the MIT License - see the LICENSE file for details.
