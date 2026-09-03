# clip_helper.py - Complete CLIP Vision Model (No API, Local)
import torch
import clip
from PIL import Image
import numpy as np

class CLIPVision:
    def __init__(self):
        """CLIP model load karo - ek baar download hoga"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📥 Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Categories for classification - CIFAR-10 + common images
        self.categories = [
            # CIFAR-10 classes
            "a cat", "a dog", "a car", "an airplane", "a ship", "a truck",
            "a bird", "a deer", "a frog", "a horse",
            # Extra categories for real world
            "a motivational poster", "text on background", "a wallpaper",
            "a person", "a human face", "a building", "a tree", "a flower",
            "food", "a book", "a phone", "a laptop", "a mountain",
            "a beach", "a sunset", "abstract art", "a logo", "a sign"
        ]
        
        # Text tokens ek baar hi banao (speed ke liye)
        self.text_tokens = clip.tokenize(self.categories).to(self.device)
        
        # Precompute text features
        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        print(f"✅ CLIP model loaded with {len(self.categories)} categories")
    
    def analyze_image(self, image):
        """Image analyze karo - object + description do"""
        try:
            # Image preprocess
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Image features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Similarity with all categories
                similarity = (image_features @ self.text_features.T).cpu().numpy()[0]
                
                # Top 3 predictions
                top3_idx = np.argsort(similarity)[-3:][::-1]
                top3_scores = similarity[top3_idx]
                top3_categories = [self.categories[i].replace("a ", "").replace("an ", "") for i in top3_idx]
                
                # Best prediction
                best_idx = top3_idx[0]
                best_score = top3_scores[0]
                best_obj = top3_categories[0]
            
            # Generate description based on top predictions
            if best_score > 0.25:  # Good confidence
                if best_score > 0.35:
                    confidence_text = "high confidence"
                else:
                    confidence_text = "medium confidence"
                
                desc = f"Image shows {best_obj} with {confidence_text} ({best_score*100:.1f}%)"
                
                # Add details from other top predictions if relevant
                if top3_scores[1] > 0.2:
                    desc += f". Could also be {top3_categories[1]}"
            else:
                best_obj = "unknown"
                desc = f"Uncertain image (top match: {top3_categories[0]} at {best_score*100:.1f}%)"
            
            return best_obj, desc
            
        except Exception as e:
            return "error", f"Analysis failed: {str(e)}"
    
    def get_top_predictions(self, image, top_k=3):
        """Top K predictions do (for display)"""
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ self.text_features.T).cpu().numpy()[0]
            
            top_idx = np.argsort(similarity)[-top_k:][::-1]
            results = []
            for idx in top_idx:
                category = self.categories[idx].replace("a ", "").replace("an ", "")
                results.append((category, float(similarity[idx])))
            
            return results
        except:
            return [("error", 0.0)] * top_k