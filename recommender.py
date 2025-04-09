import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import torch
import gc
import time
import os
import traceback
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

class SHLRecommender:
    _cache = {}
    _cache_size = 20
    def __init__(self, data_path='utils/data.csv'):
        try:
            self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}. Please check the path.")

        self.df.columns = [col.strip() for col in self.df.columns]

        # Initialize cache directory
        cache_dir = os.path.join(os.getcwd(), 'model_cache')
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using cache directory: {cache_dir}")

        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
            print("Successfully loaded all-MiniLM-L6-v2 model")
        except Exception as e:
            print(f"Error loading primary model: {str(e)}")
            try:
                # Try a different model as fallback
                print("Trying fallback model: paraphrase-MiniLM-L3-v2")
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', cache_folder=cache_dir)
                print("Successfully loaded fallback model")
            except Exception as e2:
                print(f"Error loading fallback model: {str(e2)}")
                # Create a simple embedding model as last resort
                from sentence_transformers import models, SentenceTransformer
                print("Creating basic embedding model from scratch")
                word_embedding_model = models.Transformer('bert-base-uncased', cache_dir=cache_dir)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                self.embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                print("Created basic embedding model")

        # Store cache directory for later use
        self.cache_dir = cache_dir

        # Check if Gemini API key is available
        self.use_gemini = False
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')

        if self.gemini_api_key:
            print("Gemini API key found, will use Gemini API for text generation")
            self.use_gemini = True
            # No need to initialize tokenizer and model for Gemini
            self.model = None
            self.tokenizer = None
        else:
            print("No Gemini API key found, falling back to local model")
            self._initialize_local_model()

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

    def optimize_memory(self):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._cache.clear()

        gc.collect()

        return {"status": "Memory optimized"}

    def generate_test_description(self, test_name, test_type):
        try:
            cache_key = f"{test_name}_{test_type}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            prompt = f"Write a short, factual description of '{test_name}', a {test_type} assessment, in 1-2 sentences."

            # Use Gemini API if available
            if self.use_gemini:
                try:
                    generated_text = self._generate_with_gemini(prompt)
                    # If Gemini API returned empty string (quota exceeded or error), initialize local model
                    if not generated_text and (self.model is None or self.tokenizer is None):
                        self._initialize_local_model()
                except Exception as e:
                    print(f"Error using Gemini API: {str(e)}")
                    # Fall back to local model if not initialized
                    if self.model is None or self.tokenizer is None:
                        self._initialize_local_model()
                    generated_text = ""

            # If Gemini API failed or is not available, use local model
            if not self.use_gemini or not generated_text:
                # Use local model
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, padding=True)

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=40,
                        temperature=0.2,
                        top_p=0.95,
                        do_sample=False,
                        no_repeat_ngram_size=3
                    )

                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = full_response.replace(prompt, "").strip()

            # Check if the generated text is valid
            if len(generated_text) < 20 or "write" in generated_text.lower() or "description" in generated_text.lower():
                if test_type.lower() in ["cognitive ability", "cognitive", "reasoning"]:
                    description = f"The {test_name} measures cognitive abilities and problem-solving skills."
                elif "numerical" in test_name.lower() or "numerical" in test_type.lower():
                    description = f"The {test_name} assesses numerical reasoning and data analysis abilities."
                elif "verbal" in test_name.lower() or "verbal" in test_type.lower():
                    description = f"The {test_name} evaluates verbal reasoning and language comprehension skills."
                elif "personality" in test_type.lower() or "behavioral" in test_type.lower():
                    description = f"The {test_name} assesses behavioral tendencies and personality traits in workplace contexts."
                elif "technical" in test_type.lower() or any(tech in test_name.lower() for tech in ["java", "python", ".net", "sql", "coding"]):
                    description = f"The {test_name} evaluates technical knowledge and programming skills."
                else:
                    description = f"The {test_name} assesses candidate suitability through standardized methods."
            else:
                description = generated_text

            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = description

            return description

        except Exception as e:
            print(f"Error generating description: {str(e)}")
            if test_type.lower() in ["cognitive ability", "cognitive", "reasoning"]:
                return f"The {test_name} measures cognitive abilities through structured problem-solving tasks."
            elif test_type.lower() in ["personality", "behavioral"]:
                return f"The {test_name} assesses behavioral tendencies and personality traits."
            elif "technical" in test_type.lower():
                return f"The {test_name} evaluates technical knowledge and skills."
            else:
                return f"The {test_name} assesses {test_type.lower()} capabilities."

    def _generate_with_gemini(self, prompt):
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_api_key
            }
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 100
                }
            }

            response = requests.post(url, headers=headers, json=data)

            # Check for quota exceeded or rate limit errors
            if response.status_code in [403, 429, 413]:
                print(f"API quota exceeded or rate limited: {response.status_code}")
                # Disable Gemini API for future requests in this session
                self.use_gemini = False
                # Initialize local model if not already done
                if self.model is None or self.tokenizer is None:
                    self._initialize_local_model()
                # Return empty string to trigger fallback
                return ""

            response.raise_for_status()

            result = response.json()

            # Check for error in the response
            if 'error' in result:
                error_message = result.get('error', {}).get('message', 'Unknown error')
                print(f"Gemini API error: {error_message}")
                if 'quota' in error_message.lower() or 'limit' in error_message.lower() or 'exceed' in error_message.lower():
                    self.use_gemini = False
                    if self.model is None or self.tokenizer is None:
                        self._initialize_local_model()
                return ""

            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text'].strip()

            return ""
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            error_str = str(e).lower()
            if 'quota' in error_str or 'limit' in error_str or 'exceed' in error_str or 'rate' in error_str:
                self.use_gemini = False
                if self.model is None or self.tokenizer is None:
                    self._initialize_local_model()
            return ""

    def _initialize_local_model(self):
        try:
            print("Initializing local model for text generation")
            # Initialize Qwen model
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,
                model_max_length=512,
            )

            try:
                print(f"Loading Qwen model: {model_id}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    revision="main"
                )
                print("Successfully loaded Qwen model")
            except ValueError as e:
                print(f"Error with device_map: {str(e)}")
                try:
                    print("Trying without device_map")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        cache_dir=self.cache_dir
                    )
                    print("Successfully loaded Qwen model without device_map")
                except Exception as e2:
                    print(f"Error loading Qwen model: {str(e2)}")
                    try:
                        print("Trying fallback to smaller model: distilgpt2")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            "distilgpt2",
                            cache_dir=self.cache_dir
                        )
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            "distilgpt2",
                            cache_dir=self.cache_dir
                        )
                        print("Successfully loaded fallback model")
                    except Exception as e3:
                        print(f"All model loading attempts failed: {str(e3)}")
                        print(traceback.format_exc())
                        raise ValueError("Could not load any language model. Please check your environment and permissions.")
        except Exception as e:
            print(f"Error initializing local model: {str(e)}")
            print(traceback.format_exc())
            raise

    def check_health(self):
        try:
            test_prompt = "This is a test prompt to check model health."

            start_time = time.time()
            self.embedding_model.encode(["Test embedding"])
            embedding_time = time.time() - start_time

            health_info = {
                "status": "healthy",
                "embedding_time_ms": round(embedding_time * 1000, 2),
                "cache_size": len(self._cache),
                "model_type": "gemini" if self.use_gemini else "local"
            }

            # If using Gemini API, test the API
            if self.use_gemini:
                try:
                    start_time = time.time()
                    test_result = self._generate_with_gemini("Test prompt for Gemini API health check.")
                    api_time = time.time() - start_time

                    health_info["api_time_ms"] = round(api_time * 1000, 2)
                    health_info["api_status"] = "healthy" if test_result else "unhealthy"
                except Exception as e:
                    health_info["api_status"] = "unhealthy"
                    health_info["api_error"] = str(e)
            else:
                # Test local model
                try:
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

                    health_info["tokenization_time_ms"] = round(tokenization_time * 1000, 2)
                    health_info["inference_time_ms"] = round(inference_time * 1000, 2)
                except Exception as e:
                    health_info["status"] = "unhealthy"
                    health_info["model_error"] = str(e)

            return health_info
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_recommendations(self, query, is_url=False, max_recommendations=10):
        self._cache.clear()

        if is_url:
            text = self.extract_text_from_url(query)
        else:
            text = query

        max_text_length = 2000
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."

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