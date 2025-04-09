import gradio as gr
import pandas as pd
import os

from recommender import SHLRecommender
from utils.validators import url
from fastapi import FastAPI

recommender = SHLRecommender()

def is_valid_url(input_url):
    return url(input_url)

def get_recommendations(input_text, max_recommendations):
    is_url = is_valid_url(input_text)

    # Get recommendations
    recommendations = recommender.get_recommendations(
        input_text,
        is_url=is_url,
        max_recommendations=max_recommendations
    )

    formatted_assessments = []
    for rec in recommendations:
        # Parse duration
        duration_str = rec['Duration']
        try:
            duration_int = int(''.join(filter(str.isdigit, duration_str)))
        except:
            duration_int = 60

        # Format test type as list
        test_type_list = [rec['Test Type']] if rec['Test Type'] and rec['Test Type'] != "Unknown" else ["General Assessment"]

        # Generate a proper description using the test name and type
        test_description = recommender.generate_test_description(
            test_name=rec['Test Name'],
            test_type=rec['Test Type'] if rec['Test Type'] and rec['Test Type'] != "Unknown" else "General Assessment"
        )

        # Use the generated description
        description = test_description

        url_string = rec['Link']

        formatted_assessments.append({
            "url": url_string,
            "adaptive_support": "Yes" if rec['Adaptive/IRT'] == "Yes" else "No",
            "description": description,
            "duration": duration_int,
            "remote_support": "Yes" if rec['Remote Testing'] == "Yes" else "No",
            "test_type": test_type_list
        })

    df = pd.DataFrame(formatted_assessments)

    return df

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

# Method to mount FastAPI app to Gradio for Hugging Face Spaces
def mount_gradio_app(app: FastAPI, path: str):
    """Mount a FastAPI app to a Gradio app"""
    # This is a workaround for Hugging Face Spaces
    # It allows the FastAPI app to be accessed through the Gradio app
    try:
        demo.app = app
        print(f"Successfully mounted FastAPI app to Gradio at path: {path}")
    except Exception as e:
        print(f"Error mounting FastAPI app to Gradio: {str(e)}")

if __name__ == "__main__":
    demo.launch()
