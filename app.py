from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from recommender import SHLRecommender
import uvicorn

from utils.validators import url as is_valid_url

app = FastAPI(
    title="SHL Test Recommender API",
    description="API for recommending SHL tests based on job descriptions or queries",
    version="1.0.0"
)

recommender = SHLRecommender()

# Define response models
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

@app.get("/recommend", response_model=RecommendationResponse)
async def recommend(query: str = Query(..., description="Job description text or URL")):

    try:
        is_url = is_valid_url(query)

        # Get recommendations
        recommendations = recommender.get_recommendations(
            query,
            is_url=is_url,
            max_recommendations=10
        )

        formatted_assessments = []
        for rec in recommendations:
            duration_str = rec['Duration']
            try:
                duration_int = int(''.join(filter(str.isdigit, duration_str)))
            except:
                duration_int = 60

            test_type_list = [rec['Test Type']] if rec['Test Type'] and rec['Test Type'] != "Unknown" else ["General Assessment"]

            # Generate a description based on the test name and type
            test_description = recommender.generate_test_description(
                test_name=rec['Test Name'],
                test_type=rec['Test Type'] if rec['Test Type'] and rec['Test Type'] != "Unknown" else "General Assessment"
            )

            # Use the generated description
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
