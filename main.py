import multiprocessing
import uvicorn

from app import app as fastapi_app
from ui import demo as gradio_demo

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

def run_gradio():
    gradio_demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    print("Starting SHL Recommender System...")
    
    fastapi_process = multiprocessing.Process(target=run_fastapi)
    fastapi_process.start()
    
    print("Starting Gradio interface on http://0.0.0.0:7860")
    run_gradio()
