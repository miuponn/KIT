import os
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#######################################
# 1. FASTAPI SETUP & ROUTES
#######################################

# create FastAPI app
app = FastAPI(title="KIT Fashion Assistant API")

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load env variables & set up OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    image: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    text: str
    suggestedResponses: List[str]

############################
# HELPER FUNCTIONS (used by both FastAPI & Gradio)
############################

last_response = {
    "text": "Welcome! I'm KIT, your fashion assistant. How can I help you today?",
    "suggestedResponses": []  # Empty array instead of pre-populated suggestions
}

def encode_image(image: Image.Image) -> str:
    """Encodes a PIL image to base64 PNG."""
    if image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    return ""

def generate_suggestions(user_input: str, bot_reply: str) -> List[str]:
    """
    Create context-aware follow-up suggestions based on the conversation state.
    """
    try:
        # ---- OUTFIT BUILDING MODE DETECTION WITH MORE PRECISE PATTERNS ----
        
        # First, explicitly check for all outfit building patterns
        # Including both statements and questions in outfit building mode
        
        # Look for the exact occasion question from the protocol
        is_asking_occasion = any([
            "what's the occasion for this outfit" in bot_reply.lower(),
            "what occasion are you dressing for" in bot_reply.lower(),
            bot_reply.lower().endswith("what's the occasion?"),
        ])
        
        # Check for follow-up questions in outfit building
        is_outfit_followup = any([
            # Specific protocol follow-up questions with precise wording
            "what's the vibe" in bot_reply.lower(),
            "what colour tones" in bot_reply.lower(),
            "what silhouettes" in bot_reply.lower(),
            "what's the weather" in bot_reply.lower(),
            "what's your budget" in bot_reply.lower()
        ])
        
        # Check for outfit structure
        is_outfit_structure = any(item in bot_reply for item in ["TOP:", "BOTTOM:", "FOOTWEAR:"])
        
        # Check for shopping links section with exact protocol wording
        is_shopping_links = "here are available links to source the pieces:" in bot_reply.lower()
        
        # Check for the swap question
        is_swap_question = "what would you like to change" in bot_reply.lower()
        
        # Check if asking to save the outfit
        is_save_question = "would you like to save this fit" in bot_reply.lower()
        
        # Determine if we're in outfit building mode
        is_outfit_building = any([
            is_asking_occasion,
            is_outfit_followup,
            is_outfit_structure,
            is_shopping_links,
            is_swap_question,
            is_save_question
        ])
        
        # ---- GENERATE SUGGESTIONS BASED ON DETECTED PATTERNS ----
        
        # Outfit building mode ALWAYS takes precedence, even if there are questions
        if is_outfit_building:
            if is_asking_occasion:
                return ["Wedding", "Job interview", "Casual weekend"]
                
            if is_outfit_followup:
                if "vibe" in bot_reply.lower():
                    return ["Minimalist and clean", "Edgy and avant-garde", "Relaxed but polished"]
                    
                if any(q in bot_reply.lower() for q in ["colour", "color", "tone", "palette"]):
                    return ["Monochrome black and white", "Earth tones", "Muted pastels"]
                    
                if any(q in bot_reply.lower() for q in ["silhouette", "shape", "fit"]):
                    return ["Oversized and drapey", "Slim and tailored", "Structured architectural"]
                    
                if any(q in bot_reply.lower() for q in ["weather", "temperature", "climate"]):
                    return ["Cool and rainy", "Warm summer day", "Cold winter weather"]
                    
                if any(q in bot_reply.lower() for q in ["budget", "price", "cost", "spend"]):
                    return ["Mid-range ($100-300 per piece)", "High-end designer", "Budget-friendly"]
            
            # After outfit generation
            if is_outfit_structure:
                return ["Shopping links please", "I'd like to swap some items"]
                
            # After shopping links
            if is_shopping_links:
                return ["Save this outfit for me", "I'd like to swap some items", "Add final touches"]
                
            # When asked what to change
            if is_swap_question:
                return ["Different shoes", "Something more affordable", "Different color palette"]
                
            # When asked to save the outfit
            if is_save_question:
                return ["Yes, save it", "No thanks", "Add final touches first"]
        
        # ---- QUESTION ANSWERING MODE ----
        # Check if the bot asked a question (outside outfit building mode)
        elif "?" in bot_reply:
            # Check if it's a yes/no question
            yes_no_patterns = ["do you", "would you", "could you", "are you", "will you", "have you", 
                              "is it", "should i", "can you", "does this"]
            if any(pattern in bot_reply.lower() for pattern in yes_no_patterns):
                return ["Yes.", "No."]
            
            # For other questions, generate specific answers to the question
            suggestion_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "The assistant just asked the user this question. Generate 3 brief, direct answers "
                            "(not questions) that a user might give to this specific question. Make answers very "
                            "concise (3-6 words each). Format as a simple comma-separated list without numbering."
                        )
                    },
                    {"role": "assistant", "content": bot_reply},
                ],
                max_tokens=60
            )
            suggestions_text = suggestion_response.choices[0].message.content
            suggestions = [s.strip() for s in suggestions_text.split(",") if s.strip()]
            
            # Add periods to statements if needed
            final_suggestions = []
            for suggestion in suggestions:
                suggestion = suggestion.strip()
                if suggestion and not suggestion.endswith((".", "!", "?")):
                    suggestion += "."
                final_suggestions.append(suggestion)
            
            return final_suggestions[:3]
        
        # ---- GENERAL FASHION CHAT MODE SUGGESTIONS ----
        else:
            # Check if the topic is fashion-related
            fashion_keywords = ["fashion", "style", "clothing", "outfit", "wear", "dress", 
                              "shoes", "accessory", "accessories", "trend", "designer", 
                              "brand", "garment", "fabric", "textile", "look"]
                              
            is_fashion_related = any(keyword in (user_input + " " + bot_reply).lower() for keyword in fashion_keywords)
            
            if not is_fashion_related:
                # Not fashion related - provide redirection suggestions
                return [
                    "What's trending in fashion?",
                    "I'd like to build another outfit.",
                    "Can you suggest a style for me?"
                ]
            
            # Generate context-aware fashion suggestions
            suggestion_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fashion assistant. Based on this conversation about fashion, generate 3 "
                            "brief follow-up prompts or questions (5-8 words each) the user might request next. "
                            "These should be directly related to the fashion topic just discussed and encourage "
                            "exploration of related fashion concepts. Format as a simple comma-separated list."
                        )
                    },
                    {"role": "user", "content": user_input or "Fashion advice"},
                    {"role": "assistant", "content": bot_reply},
                ],
                max_tokens=60
            )
            suggestions_text = suggestion_response.choices[0].message.content
            suggestions = [s.strip() for s in suggestions_text.split(",") if s.strip()]
            
            # Before returning suggestions, add periods to statements
            final_suggestions = []
            for suggestion in suggestions:
                suggestion = suggestion.strip()
                if suggestion and not suggestion.endswith((".", "!", "?")):
                    suggestion += "."
                final_suggestions.append(suggestion)
            
            return final_suggestions[:3]
        
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return ["Try a fashion question.", "Style me for an event.", "What's trending now?"]

def generate_contextual_followups(chat_history: List[Dict[str, Any]]) -> List[str]:
    """
    Generate follow-up suggestions based on the entire conversation context.
    """
    if not chat_history:
        return ["What's trending now?", "Style advice please", "Recommended brands?"]
    try:
        # use last ~6 messages for context
        recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
        messages = []
        for msg in recent_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

        suggestion_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Based on this fashion conversation history, generate 3 brief follow-up questions "
                        "(3-5 words each) that the user might want to ask next. Format as a simple comma-separated list."
                    )
                },
                *messages
            ],
            max_tokens=50
        )
        suggestions_text = suggestion_response.choices[0].message.content
        suggestions = [s.strip() for s in suggestions_text.split(",") if s.strip()]
        return suggestions[:3]
    except Exception as e:
        print(f"Error generating contextual follow-ups: {e}")
        return ["Tell me more", "Other options?", "Any alternatives?"]

# helper function to ensure 3 suggestions aren't always needed
def ensure_suggestions(suggestions_list, max_count=3):
    """Return up to max_count suggestions, without padding if fewer are provided."""
    return suggestions_list[:max_count] if suggestions_list else []

# Add this function to determine if suggestions are needed
def should_show_suggestions(bot_reply: str) -> bool:
    """Determine if we should show suggestions based on context."""
    # Don't show suggestions for explanations without questions
    if len(bot_reply) > 100 and "?" not in bot_reply:
        return False
        
    # Other cases where we might want to skip suggestions
    skip_patterns = [
        "anything else you'd like to know",
        "hope that helps",
        "let me know if you need"
    ]
    
    if any(pattern in bot_reply.lower() for pattern in skip_patterns):
        return False
        
    return True

#######################################
# 2. FASTAPI ENDPOINTS
#######################################

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy", "service": "KIT Fashion Assistant"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(request: ChatRequest):
    """
    Main chat endpoint for the KIT fashion assistant.
    - message (str): user text
    - image (str, optional): base64-encoded image
    - history (list, optional): chat history with role & content
    Returns: { text: str, suggestedResponses: str[] }
    """
    message = request.message
    image_data = request.image
    chat_history = request.history or []

    # prepare system prompt
    system_message = (
            "You are KIT: a helpful AI assistant for PPPTAILORINGCOURIER, "
            "focused on answering questions about archive and avant-garde fashion, "
            "as well as personal style. Your responses should be concise, thoughtful, "
            "and reflect the brand's minimalist, futuristic aesthetic. "
            "Keep responses under 3 sentences when possible, and be concise when asking questions. "
            "You can also process images of clothing and provide feedback.\n\n"
            
            "OUTFIT BUILDING PROTOCOL:\n"
            "When a user asks you to build or suggest an outfit/fit/look:\n"
            "1. If no occasion is specified, ALWAYS ask 'What's the occasion for this outfit?'\n"
            "2. If the occasion IS specified, ask 2 of these follow-up questions (one at a time) with a brief affirmation preceding it (ie. Nice. Got it. etc):\n"
            "   - What's the vibe?\n"
            "   - What colour tones do you prefer?\n"
            "   - What silhouettes do you prefer?\n"
            "   - What's the weather going to be like?\n"
            "   - What's your budget like?\n"
            "3. Then structure your outfit recommendation as follows:\n\n"
            
            "[OUTFIT NAME]\n"
            "- TOP: [Brief title of garment] - [Price range] - ([Designer or brand])\n"
            "- BOTTOM: [Brief title of garment] - [Price range] - ([Designer or brand])\n"
            "- FOOTWEAR: [Brief title of shoe] - [Price range] - ([Designer or brand])\n"
            "- ACCESSORIES: [List 1-2 key accessories] \n\n"
            
            "4. When the user asks for shopping links, provide them in this format:\n"
            "Here are available links to source the pieces:\n"
            "- [Brief item title]: [Brand] ([Online shop])\n"
            "- [Brief item title]: [Brand] ([Online shop])\n"
            "- [Brief item title]: [Brand] ([Online shop])\n"
            "Would you like to save this fit to your style archive?\n\n"
            
            "5. If the user asks to swap items but doesn't specify what to swap, ask 'What would you like to change?'\n"
            "6. After 1-2 follow-up questions about swap details, regenerate the outfit list.\n\n"
            
            "7. When the user requests final touches, provide them as follows:\n"
            "## FINAL TOUCHES\n"
            "- FRAGRANCE: [Specific fragrance] - [Brief description of scent profile]\n"
            "- ADDITIONAL ACCESSORIES: [1-2 subtle additions] - [Brief styling tip]\n"
            "- POSTURE/STYLING: [Brief suggestion on how to carry/wear the look]\n\n"
            
            "Suggest specific archive or contemporary designer pieces when appropriate. "
            "For each recommendation, focus on avant-garde, minimalist, or conceptual fashion houses "
            "like Maison Margiela, Rick Owens, Comme des Garçons, Jil Sander, Lemaire, etc."
            "But otherwise if the budget specified is lower, look for affordable but fashion-community approved staples like Uniqlo, Nike, etc."
            
            "IMAGE ANALYSIS PROTOCOL:\n"
            "When a user uploads an image:\n"
            "1. If in outfit building mode, analyze the image and continue with the current stage\n"
            "2. If not in outfit building mode, analyze the image and provide fashion insights\n"
            "3. For clothing images, identify: style, brand (if recognizable), key details, and styling suggestions\n"
            "4. For outfit images, comment on overall look, coordination, and potential improvements\n"
        )

    # Build the conversation
    messages = [{"role": "system", "content": system_message}]
    for msg in chat_history:
        if "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # if there's an image, parse it
    if image_data and image_data.startswith("data:image"):
        try:
            image_bytes = base64.b64decode(image_data.split(",")[1])
            pil_img = Image.open(BytesIO(image_bytes))
            # add the image to the conversation
            base64_img = encode_image(pil_img)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message or "Please analyze this fashion image."},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img}"}
                ],
            })
        except Exception as e:
            print("Error decoding image:", e)
            messages.append({"role": "user", "content": message or "Image upload failed."})
    else:
        # otherwise just add the text message
        messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
        )
        bot_reply = response.choices[0].message.content
        # generate follow-up suggestions
        suggestions = generate_suggestions(message, bot_reply) if should_show_suggestions(bot_reply) else []

        return {"text": bot_reply, "suggestedResponses": suggestions}
    except Exception as e:
        print("Error in /api/chat:", e)
        return {
            "text": "I'm sorry, I encountered an error processing your request.",
            "suggestedResponses": ["Try a different question", "Style advice please", "What's trending?"]
        }

@app.get("/api/starter-prompts")
async def get_starter_prompts():
    """
    Returns suggested starter prompts for chat cards.
    """
    return {
        "prompts": [
            "What should I wear to a gallery opening?",
            "Suggest styling tips for minimalist wardrobe",
            "Break down a trend: tabis"
        ]
    }

@app.post("/api/suggested-responses")
async def get_suggested_responses(data: dict):
    """
    Generate suggested responses based on conversation context or (message + bot_reply).
    """
    message = data.get("message", "")
    bot_reply = data.get("bot_reply", "")
    history = data.get("history", [])

    try:
        # if both a user message & bot reply exist, do direct suggestions
        if message and bot_reply:
            suggestions = generate_suggestions(message, bot_reply)
            return {"suggestedResponses": suggestions}

        # if we have history, do contextual follow-ups
        elif history:
            # convert any 2-length lists to {role:..., content:...} as needed
            formatted_history = []
            for item in history:
                if isinstance(item, dict) and "role" in item:
                    formatted_history.append(item)
                elif isinstance(item, list) and len(item) == 2:
                    formatted_history.append({"role": "user", "content": item[0]})
                    formatted_history.append({"role": "assistant", "content": item[1]})

            suggestions = generate_contextual_followups(formatted_history)
            return {"suggestedResponses": suggestions}

        # otherwise return the last known responses or some default
        global last_response
        return {"suggestedResponses": last_response["suggestedResponses"]}

    except Exception as e:
        print("Error in /api/suggested-responses:", e)
        return {"suggestedResponses": ["Try a different question", "Style advice please", "What's trending?"]}

@app.get("/api/last-response")
async def get_last_response():
    """Return the last stored bot response and suggested follow-ups."""
    global last_response
    return last_response

@app.get("/api/suggestions")
async def get_suggestions(message: str = "", bot_reply: str = ""):
    """
    Simple endpoint that returns suggestions for a given message and bot reply.
    """
    try:
        # Only generate suggestions if we have both message and bot_reply
        if not message or not bot_reply:
            return {"suggestions": []}
            
        # Generate suggestions using both user message and bot reply
        suggestions = generate_suggestions(message, bot_reply)
        return {"suggestions": suggestions}
    except Exception as e:
        print("Error in /api/suggestions:", e)
        return {"suggestions": []}

@app.post("/api/analyze-image")
async def analyze_image(data: dict):
    """
    Analyze a fashion image and return insights.
    """
    image_data = data.get("image", "")
    context = data.get("context", "")  # Optional context about what user wants to know
    
    if not image_data or not image_data.startswith("data:image"):
        return {"error": "Valid image data required"}
        
    try:
        # Decode the image
        image_bytes = base64.b64decode(image_data.split(",")[1])
        pil_img = Image.open(BytesIO(image_bytes))
        base64_img = encode_image(pil_img)
        
        # Create a message with the image
        messages = [
            {"role": "system", "content": "You are a fashion expert analyzing clothing images. Focus on style, cut, fabric, design elements, and potential brand identification. Keep your analysis concise."},
            {"role": "user", "content": [
                {"type": "text", "text": context or "Please analyze this fashion item."},
                {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img}"}
            ]}
        ]
        
        # Get analysis from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=250,
        )
        
        analysis = response.choices[0].message.content
        suggestions = generate_suggestions("", analysis) if should_show_suggestions(analysis) else []
        
        return {
            "analysis": analysis,
            "suggestedResponses": suggestions
        }
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return {"error": "Failed to analyze image", "details": str(e)}

@app.get("/api/docs/image-upload")
async def image_upload_docs():
    """Documentation for image upload functionality"""
    return {
        "formats": ["image/jpeg", "image/png"],
        "maxSize": "5MB",
        "encoding": "base64",
        "example": {
            "endpoint": "/api/chat",
            "method": "POST",
            "payload": {
                "message": "What do you think of this outfit?",
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
                "history": []
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)