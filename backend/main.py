import os
import uvicorn

from app import app

if __name__ == "__main__":
    # Check if running on Hugging Face Spaces
    IS_HF_SPACE = os.environ.get('SPACE_ID') is not None
    port = 7860 if IS_HF_SPACE else 8000
    
    print(f"Starting SHL Test Recommender API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
