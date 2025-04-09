import os
import uvicorn
import threading
import time

from app import app as fastapi_app
from ui import demo as gradio_demo

# Check if running on Hugging Face Spaces
IS_HF_SPACE = os.environ.get('SPACE_ID') is not None

def run_fastapi():
    # On Hugging Face Spaces, we need to use a different port
    port = 7860 if IS_HF_SPACE else 8000
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)

def run_gradio():
    # On Hugging Face Spaces, we need to mount the FastAPI app
    if IS_HF_SPACE:
        # For Hugging Face Spaces, mount the FastAPI app to the Gradio app
        gradio_demo.mount_gradio_app(fastapi_app, "/api")
        gradio_demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    else:
        # For local development, run Gradio on port 7860
        gradio_demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    print("Starting SHL Recommender System...")

    if IS_HF_SPACE:
        print("Running on Hugging Face Spaces - using integrated server")
        # On Hugging Face Spaces, we run Gradio with the FastAPI app mounted
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
