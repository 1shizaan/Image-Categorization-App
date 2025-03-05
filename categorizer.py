# categorizer.py
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import shutil
import json
import logging
from typing import List, Dict, Union
import glob
import numpy as np

import streamlit as st  # You may or may not need Streamlit inside here, depending on usage

class ImageCategorizer:
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32", 
        config_path: str = "domain_config.json"
    ):
        """
        Initialize the Image Categorizer with CLIP model and configuration.
        """
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.config_path = config_path
        self.domains = self.load_or_create_config()
        
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Error loading CLIP model: {e}")
            raise
        
        self.output_base_dir = "categorized_images"
        os.makedirs(self.output_base_dir, exist_ok=True)
    
    def load_or_create_config(self) -> Dict[str, Dict]:
        """
        Load or create domain configuration with more detailed descriptions.
        """
        default_domains = {
            "Technology": {
                "descriptions": [
                    "a screenshot of a computer interface", 
                    "software development environment", 
                    "programming code editor", 
                    "tech dashboard", 
                    "computer settings", 
                    "software application"
                ]
            },
            "Finance": {
                "descriptions": [
                    "financial dashboard", 
                    "banking website", 
                    "stock market interface", 
                    "investment platform", 
                    "online banking screen", 
                    "cryptocurrency exchange"
                ]
            },
            "Health": {
                "descriptions": [
                    "medical application", 
                    "health tracking screen", 
                    "fitness app", 
                    "medical records interface", 
                    "telemedicine platform", 
                    "health monitoring dashboard"
                ]
            },
            "Education": {
                "descriptions": [
                    "online learning platform", 
                    "educational website", 
                    "study materials screen", 
                    "course management system", 
                    "online classroom interface", 
                    "educational resource page"
                ]
            },
            "Travel": {
                "descriptions": [
                    "travel booking website", 
                    "maps application", 
                    "travel planning interface", 
                    "airline booking screen", 
                    "hotel reservation page", 
                    "navigation application"
                ]
            },
            "Entertainment": {
                "descriptions": [
                    "streaming platform", 
                    "game interface", 
                    "media website", 
                    "video platform", 
                    "entertainment application", 
                    "movie or TV show screen"
                ]
            },
            "E-commerce": {
                "descriptions": [
                    "online shopping website", 
                    "product page", 
                    "marketplace interface", 
                    "shopping cart screen", 
                    "product catalog", 
                    "online store dashboard"
                ]
            },
            "Social Media": {
                "descriptions": [
                    "social network interface", 
                    "messaging platform", 
                    "social media feed", 
                    "chat application", 
                    "profile page", 
                    "social network dashboard"
                ]
            },
            "News": {
                "descriptions": [
                    "news website", 
                    "online newspaper", 
                    "news article page", 
                    "news feed", 
                    "current events portal", 
                    "media news interface"
                ]
            },
            "Productivity": {
                "descriptions": [
                    "task management app", 
                    "project tracking interface", 
                    "productivity tool", 
                    "to-do list", 
                    "workflow management", 
                    "collaboration platform"
                ]
            }
        }
        
        # Try to load or save configuration
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_domains, f, indent=4)
        except Exception as e:
            self.logger.error(f"Could not save default config: {e}")
        
        return default_domains
    
    def categorize_screenshot(self, image_path: str) -> Dict[str, Union[str, float]]:
        """
        Categorize a screenshot using CLIP zero-shot classification with detailed probabilities.
        """
        try:
            # Prepare all possible descriptions
            all_descriptions = []
            domain_names = []
            for domain, config in self.domains.items():
                for desc in config.get("descriptions", []):
                    all_descriptions.append(f"a screenshot related to {domain.lower()}: {desc}")
                    domain_names.append(domain)
            
            # Add a generic "other" category
            all_descriptions.append("a screenshot not related to any specific category")
            domain_names.append("Other")
            
            # Process image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                text=all_descriptions, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
            # Convert to numpy for easier manipulation
            probs_np = probs.numpy()
            
            # Get top predictions
            top_indices = probs_np.argsort()[::-1][:3]
            top_domains = [domain_names[idx] for idx in top_indices]
            top_probs = probs_np[top_indices]
            
            # Detailed logging
            for domain, prob in zip(top_domains, top_probs):
                self.logger.info(f"Domain: {domain}, Probability: {prob:.2%}")
            
            return {
                "domain": top_domains[0],
                "probability": float(top_probs[0]),
                "top_domains": top_domains,
                "probabilities": [float(p) for p in top_probs]
            }
        
        except FileNotFoundError:
            self.logger.error(f"File not found: {image_path}")
            return {"domain": "Error", "probability": 0.0, "top_domains": [], "probabilities": []}
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return {"domain": "Error", "probability": 0.0, "top_domains": [], "probabilities": []}
    
    def save_categorized_image(self, image_path: str, domain: str) -> str:
        """
        Save image to its corresponding domain folder.
        """
        try:
            # Create domain-specific folder if it doesn't exist
            domain_folder = os.path.join(self.output_base_dir, domain)
            os.makedirs(domain_folder, exist_ok=True)
            
            # Generate unique filename
            filename = os.path.basename(image_path)
            unique_filename = filename
            counter = 1
            while os.path.exists(os.path.join(domain_folder, unique_filename)):
                name, ext = os.path.splitext(filename)
                unique_filename = f"{name}_{counter}{ext}"
                counter += 1
            
            # Copy image to domain folder
            destination = os.path.join(domain_folder, unique_filename)
            shutil.copy2(image_path, destination)
            
            self.logger.info(f"Saved {filename} to {destination}")
            return destination
        
        except Exception as e:
            self.logger.error(f"Error saving {image_path}: {e}")
            return ""
    
    def process_images(self, image_paths: List[str]) -> Dict[str, Dict[str, Union[str, bool]]]:
        """
        Process multiple images, categorize and save them.
        """
        results = {}
        for path in image_paths:
            categorization = self.categorize_screenshot(path)
            
            # Only save if domain is not 'Error' and probability is above a threshold
            if categorization['domain'] != "Error" and categorization['probability'] > 0.1:
                saved_path = self.save_categorized_image(path, categorization['domain'])
                results[path] = {
                    "domain": categorization['domain'],
                    "probability": categorization['probability'],
                    "top_domains": categorization['top_domains'],
                    "saved": bool(saved_path),
                    "saved_path": saved_path
                }
            else:
                results[path] = {
                    "domain": categorization['domain'],
                    "probability": categorization['probability'],
                    "top_domains": categorization['top_domains'],
                    "saved": False,
                    "saved_path": ""
                }
        return results
