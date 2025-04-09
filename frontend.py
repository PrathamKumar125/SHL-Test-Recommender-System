import gradio as gr
import pandas as pd
import os
import sys
import requests
import json

# Add parent directory to path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validators import url

# Configuration for backend API
BACKEND_API_URL = "https://pratham0011-shl-test-recommender-api.hf.space"

def is_valid_url(input_url):
    return url(input_url)

def get_recommendations(input_text, max_recommendations):
    try:
        is_url = is_valid_url(input_text)
        
        # Make API request to backend
        api_url = f"{BACKEND_API_URL}/recommend"
        payload = {
            "query": input_text,
            "max_recommendations": max_recommendations
        }
        
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Convert to DataFrame for Gradio display
        formatted_assessments = []
        for assessment in data.get("recommended_assessments", []):
            formatted_assessments.append({
                "url": assessment.get("url", ""),
                "adaptive_support": assessment.get("adaptive_support", "No"),
                "description": assessment.get("description", ""),
                "duration": assessment.get("duration", 60),
                "remote_support": assessment.get("remote_support", "No"),
                "test_type": ", ".join(assessment.get("test_type", ["General Assessment"]))
            })
        
        df = pd.DataFrame(formatted_assessments)
        return df
    except Exception as e:
        # Return error as DataFrame for display
        return pd.DataFrame([{"url": "", "adaptive_support": "", "description": f"Error: {str(e)}", "duration": 0, "remote_support": "", "test_type": ""}])

with gr.Blocks(title="SHL Test Recommender") as demo:
    gr.Markdown("# SHL Test Recommender")
    gr.Markdown("""
    This tool recommends SHL tests based on job descriptions or natural language queries.
    """)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter job description or URL",
                placeholder="Paste job description or URL here...",
                lines=10
            )
            max_recommendations = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                label="Maximum number of recommendations"
            )
            submit_btn = gr.Button("Get Recommendations", variant="primary")

    recommendations_output = gr.DataFrame(
        label="Recommended SHL Tests",
        headers=["url", "adaptive_support", "description", "duration", "remote_support", "test_type"],
        interactive=False
    )

    submit_btn.click(
        fn=get_recommendations,
        inputs=[input_text, max_recommendations],
        outputs=[recommendations_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
