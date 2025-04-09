import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import torch
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

class SHLRecommender:
    _cache = {}
    _cache_size = 20
    def __init__(self, data_path='utils/data.csv'):
        try:
            self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}. Please check the path.")

        # Clean column names
        self.df.columns = [col.strip() for col in self.df.columns]

        os.environ['TRANSFORMERS_CACHE'] = './cache'
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
            model_max_length=512,
        )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        except ValueError as e:
            raise ValueError(f"Model loading failed: {str(e)}. Ensure you have enough memory or try a different model.")

        self.create_embeddings()

    def create_embeddings(self):
        texts = []
        for _, row in self.df.iterrows():
            text = f"{row['Test Name']} {row['Test Type']}"
            texts.append(text)

        self.product_embeddings = self.embedding_model.encode(texts)

    def extract_text_from_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            return f"Error extracting text from URL: {str(e)}"

    def analyze_job_description(self, text):
        max_text_length = 1000
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."

        prompt = f"""
        Analyze this job description and identify key skills and suitable assessment types.

        Job Description: {text}

        Instructions:
        1. Identify if the job requires: Cognitive abilities, Personality traits, Technical skills, Situational judgment, or Job-specific knowledge
        2. Provide a brief summary (3-5 sentences) of the key requirements
        3. DO NOT use code, functions, or programming syntax in your response
        4. Format your response as plain text only
        5. Keep your response concise and focused on assessment recommendations
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Instructions:" in analysis:
            response = analysis.split("Instructions:")[-1].strip()
        else:
            response = analysis.split("Job Description:")[-1].strip()

        response = response.replace("function", "")
        response = response.replace("{", "")
        response = response.replace("}", "")
        response = response.replace(";", "")
        response = response.replace("//", "")
        response = response.replace("/*", "")
        response = response.replace("*/", "")


        return response

    def optimize_memory(self):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._cache.clear()

        gc.collect()

        return {"status": "Memory optimized"}

    def generate_test_description(self, test_name, test_type):
        """Generate a concise, factual description for an SHL test based on its name and type"""
        try:
            # Use a more structured prompt with explicit constraints to avoid hallucinations
            prompt = f"""
            You are an SHL assessment expert. Create a brief, factual description of the SHL assessment named "{test_name}" which is categorized as "{test_type}".

            Rules:
            1. Focus ONLY on what this specific assessment likely measures based on its name and type
            2. Keep your description to 1-2 short, factual sentences
            3. DO NOT mention job descriptions, specific positions, or make claims about effectiveness
            4. DO NOT use phrases like "This assessment is designed to" or "This test helps"
            5. DO NOT use technical jargon, code syntax, or numbered lists
            6. DO NOT repeat the instructions or include "Instructions:" in your response
            7. Start directly with the description

            Example good response: "The Numerical Reasoning assessment measures a candidate's ability to analyze and interpret numerical data and make logical decisions."
            """

            # Generate the description with stricter parameters
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=True)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,  # Even shorter description
                    temperature=0.9,  # Lower temperature for more predictable output
                    top_p=0.85,
                    do_sample=False,  # Deterministic generation
                    no_repeat_ngram_size=3  # Prevent repetition
                )

            # Decode and clean up the description
            description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt and any instructions
            if "Rules:" in description:
                description = description.split("Rules:")[0].strip()
            if "Example good response:" in description:
                description = description.split("Example good response:")[1].strip()
            if "Instructions:" in description:
                description = description.split("Instructions:")[0].strip()

            # Remove quotes if present
            description = description.strip('"').strip()

            # Remove any remaining prompt text
            description = description.replace(prompt.strip(), "").strip()

            # Additional cleaning
            description = description.replace("function", "")
            description = description.replace("{", "")
            description = description.replace("}", "")
            description = description.replace(";", "")
            description = description.replace("1.", "")
            description = description.replace("2.", "")
            description = description.replace("3.", "")

            # If the description is still problematic, use a template
            if len(description) < 20 or "job description" in description.lower() or "this assessment is designed" in description.lower():
                if test_type.lower() in ["cognitive ability", "cognitive", "reasoning"]:
                    description = f"The {test_name} measures a candidate's {test_type.lower()} through structured problem-solving tasks."
                elif test_type.lower() in ["personality", "behavioral"]:
                    description = f"The {test_name} assesses a candidate's behavioral tendencies and personality traits relevant to workplace performance."
                elif "technical" in test_type.lower():
                    description = f"The {test_name} evaluates a candidate's technical knowledge and skills in {test_type.lower().replace('technical', '').strip()} areas."
                else:
                    description = f"The {test_name} assesses a candidate's {test_type.lower()} capabilities through standardized testing methods."

            return description

        except Exception:
            # Return a template-based description if there's an error
            if test_type.lower() in ["cognitive ability", "cognitive", "reasoning"]:
                return f"The {test_name} measures a candidate's cognitive abilities through structured problem-solving tasks."
            elif test_type.lower() in ["personality", "behavioral"]:
                return f"The {test_name} assesses a candidate's behavioral tendencies and personality traits."
            elif "technical" in test_type.lower():
                return f"The {test_name} evaluates a candidate's technical knowledge and skills."
            else:
                return f"The {test_name} assesses a candidate's {test_type.lower()} capabilities."

    def check_health(self):
        try:
            test_prompt = "This is a test prompt to check model health."

            start_time = time.time()
            inputs = self.tokenizer(
                test_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=32,
                padding=True
            )
            tokenization_time = time.time() - start_time

            start_time = time.time()
            with torch.no_grad():
                _ = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=20,
                    do_sample=True
                )
            inference_time = time.time() - start_time

            start_time = time.time()
            self.embedding_model.encode(["Test embedding"])
            embedding_time = time.time() - start_time

            return {
                "status": "healthy",
                "tokenization_time_ms": round(tokenization_time * 1000, 2),
                "inference_time_ms": round(inference_time * 1000, 2),
                "embedding_time_ms": round(embedding_time * 1000, 2),
                "cache_size": len(self._cache)
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_recommendations(self, query, is_url=False, max_recommendations=10):
        # Clear cache after every request as requested
        self._cache.clear()

        if is_url:
            text = self.extract_text_from_url(query)
        else:
            text = query

        max_text_length = 2000
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."

        # Get embeddings directly from the text without using job analysis
        # This prevents hallucination in job descriptions
        query_embedding = self.embedding_model.encode(text[:1000])

        similarity_scores = cosine_similarity(
            [query_embedding],
            self.product_embeddings
        )[0]

        top_indices = np.argsort(similarity_scores)[::-1][:max_recommendations]

        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'Test Name': self.df.iloc[idx]['Test Name'],
                'Test Type': self.df.iloc[idx]['Test Type'],
                'Remote Testing': self.df.iloc[idx]['Remote Testing (Yes/No)'],
                'Adaptive/IRT': self.df.iloc[idx]['Adaptive/IRT (Yes/No)'],
                'Duration': self.df.iloc[idx]['Duration'],
                'Link': self.df.iloc[idx]['Link']
            })

        return recommendations