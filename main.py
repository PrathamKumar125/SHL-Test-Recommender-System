import os
import uvicorn
import threading
import time

from app import app as fastapi_app
from ui import demo as gradio_demo

# Check if running on Hugging Face Spaces
IS_HF_SPACE = os.environ.get('SPACE_ID') is not None

def run_fastapi():
    # Run FastAPI on port 8000 for local development
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

def run_gradio():
    # For Hugging Face Spaces, we need to use the FastAPI app directly
    if IS_HF_SPACE:
        # On Hugging Face Spaces, Gradio will use the FastAPI app
        # This makes the FastAPI endpoints available at /api/...
        print("Running on Hugging Face Spaces - using FastAPI integration")
        # Updated integration method - use .queue() to ensure app is initialized properly
        from fastapi.middleware.cors import CORSMiddleware
        import gradio as gr
        
        # Configure CORS for the FastAPI application
        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount the Gradio app to FastAPI
        gradio_app = gr.mount_gradio_app(
            app=fastapi_app,
            blocks=gradio_demo,
            path="/gradio"
        )
        
        # Start the FastAPI server
        uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
    else:
        # For local development, just run Gradio normally
        gradio_demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    print("Starting SHL Recommender System...")

    if IS_HF_SPACE:
        print("Running on Hugging Face Spaces - using integrated server")
        # On Hugging Face Spaces, we run Gradio with the FastAPI app
        run_gradio()
    else:
        print("Running locally - starting separate servers")
        # Start FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()

        # Give FastAPI time to start
        time.sleep(2)

        print("Starting Gradio interface on http://0.0.0.0:7860")
        print("FastAPI running on http://0.0.0.0:8000")
        
        # Run Gradio in the main thread
        run_gradio()
