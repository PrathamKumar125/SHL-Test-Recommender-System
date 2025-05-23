from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import SHLRecommender
from utils.validators import url as is_valid_url

app = FastAPI(
    title="SHL Test Recommender API",
    description="API for recommending SHL tests based on job descriptions or queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow requests from any origin
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

recommender = SHLRecommender()

# Define request and response models
class RecommendRequest(BaseModel):
    query: str
    max_recommendations: int = 10

class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]

# API endpoints
@app.get("/health")
async def health_check():
    try:
        if not recommender or not hasattr(recommender, 'df') or recommender.df.empty:
            return {"status": "unhealthy"}

        if not hasattr(recommender, 'embedding_model') or not hasattr(recommender, 'model') or not hasattr(recommender, 'tokenizer'):
            return {"status": "unhealthy"}

        if not hasattr(recommender, 'product_embeddings') or len(recommender.product_embeddings) == 0:
            return {"status": "unhealthy"}

        return {"status": "healthy"}
    except Exception:
        return {"status": "unhealthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to the SHL Test Recommender API."}

@app.post("/optimize")
async def optimize_memory():
    try:
        recommender.optimize_memory()
        return {"status": "success", "message": "Memory optimized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main recommend endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendRequest):
    return await process_recommendation(request.query, request.max_recommendations)


async def process_recommendation(query: str, max_recommendations: int):
    try:
        is_url = is_valid_url(query)

        recommendations = recommender.get_recommendations(
            query,
            is_url=is_url,
            max_recommendations=max_recommendations
        )

        formatted_assessments = []
        for rec in recommendations:
            duration_str = rec['Duration']
            try:
                duration_int = int(''.join(filter(str.isdigit, duration_str)))
            except:
                duration_int = 60

            test_type_list = [rec['Test Type']] if rec['Test Type'] and rec['Test Type'] != "Unknown" else ["General Assessment"]

            test_description = recommender.generate_test_description(
                test_name=rec['Test Name'],
                test_type=rec['Test Type'] if rec['Test Type'] and rec['Test Type'] != "Unknown" else "General Assessment"
            )

            description = test_description

            formatted_assessments.append(
                Assessment(
                    url=rec['Link'],
                    adaptive_support="Yes" if rec['Adaptive/IRT'] == "Yes" else "No",
                    description=description,
                    duration=duration_int,
                    remote_support="Yes" if rec['Remote Testing'] == "Yes" else "No",
                    test_type=test_type_list
                )
            )

        return RecommendationResponse(
            recommended_assessments=formatted_assessments
        )
    except Exception as e:
        try:
            recommender.optimize_memory()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Check if running on Hugging Face Spaces
    IS_HF_SPACE = os.environ.get('SPACE_ID') is not None
    port = 7860 if IS_HF_SPACE else 8000
    
    print(f"Starting FastAPI server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
