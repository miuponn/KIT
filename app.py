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

# Create FastAPI app
app = FastAPI(title="KIT Fashion Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables & set up OpenAI
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
    "suggestedResponses": ["What's trending now?", "Style advice please", "Tell me about tabis"]
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
    Create 3 short follow-up suggestions (3-5 words each) using 'gpt-4o'.
    Returns a list of up to 3 suggestions.
    """
    try:
        suggestion_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate 3 very brief follow-up questions (3-5 words each) "
                        "related to fashion and the previous conversation. "
                        "Format as a simple comma-separated list without numbering or quotes."
                    )
                },
                {"role": "user", "content": user_input or "Fashion image analysis"},
                {"role": "assistant", "content": bot_reply},
            ],
            max_tokens=50
        )
        suggestions_text = suggestion_response.choices[0].message.content
        suggestions = [s.strip() for s in suggestions_text.split(",") if s.strip()]
        return suggestions[:3]
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return ["Try a different question", "Style me again", "Any new trends?"]

def generate_contextual_followups(chat_history: List[Dict[str, Any]]) -> List[str]:
    """
    Generate follow-up suggestions based on the entire conversation context.
    """
    if not chat_history:
        return ["What's trending now?", "Style advice please", "Tell me about tabis"]
    try:
        # Use last ~6 messages for context
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

    # Prepare system prompt
    system_message = (
            "You are KIT: a helpful AI assistant for PPPTAILORINGCOURIER, "
            "focused on answering questions about archive and avant-garde fashion, "
            "as well as personal style. Your responses should be concise, thoughtful, "
            "and reflect the brand's minimalist, futuristic aesthetic. "
            "Keep responses under 3 sentences when possible. "
            "You can also process images of clothing and provide feedback.\n\n"
            
            "OUTFIT BUILDING PROTOCOL:\n"
            "When a user asks you to build or suggest an outfit/fit/look:\n"
            "1. First, ask follow-up questions about the specific context (formal event, casual outing, work, specific weather).\n"
            "2. Ask about their style preferences (minimalist, avant-garde, etc.).\n"
            "3. Then structure your outfit recommendation as follows:\n\n"
            
            "## [OUTFIT NAME]\n"
            "- TOP: [Specific garment with designer] - [Price range] - [Brief description]\n"
            "- BOTTOM: [Specific garment with designer] - [Price range] - [Brief description]\n"
            "- FOOTWEAR: [Specific shoe with designer] - [Price range] - [Brief description]\n"
            "- ACCESSORIES: [List 1-2 key accessories] - [Price range]\n\n"
            
            "4. When the user expresses satisfaction with the outfit, ask if they would like to save it to their archive.\n"
            "5. After their response (regardless of yes/no), offer final touches as follows:\n\n"
            
            "## FINAL TOUCHES\n"
            "- FRAGRANCE: [Specific fragrance] - [Brief description of scent profile]\n"
            "- ADDITIONAL ACCESSORIES: [1-2 subtle additions] - [Brief styling tip]\n"
            "- POSTURE/STYLING: [Brief suggestion on how to carry/wear the look]\n\n"
            
            "Suggest specific archive or contemporary designer pieces when appropriate. "
            "For each recommendation, focus on avant-garde, minimalist, or conceptual fashion houses "
            "like Maison Margiela, Rick Owens, Comme des Garçons, Jil Sander, Lemaire, etc."
        )

    # Build the conversation
    messages = [{"role": "system", "content": system_message}]
    for msg in chat_history:
        if "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # If there's an image, parse it
    if image_data and image_data.startswith("data:image"):
        try:
            image_bytes = base64.b64decode(image_data.split(",")[1])
            pil_img = Image.open(BytesIO(image_bytes))
            # Add the image to the conversation
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
        # Otherwise just add the text message
        messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=250,
        )
        bot_reply = response.choices[0].message.content
        # Generate follow-up suggestions
        suggestions = generate_suggestions(message, bot_reply)

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
            "Tell me about tabi boots",
            "What are good fabrics for summer?"
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
        # If both a user message & bot reply exist, do direct suggestions
        if message and bot_reply:
            suggestions = generate_suggestions(message, bot_reply)
            return {"suggestedResponses": suggestions}

        # If we have history, do contextual follow-ups
        elif history:
            # Convert any 2-length lists to {role:..., content:...} as needed
            formatted_history = []
            for item in history:
                if isinstance(item, dict) and "role" in item:
                    formatted_history.append(item)
                elif isinstance(item, list) and len(item) == 2:
                    formatted_history.append({"role": "user", "content": item[0]})
                    formatted_history.append({"role": "assistant", "content": item[1]})

            suggestions = generate_contextual_followups(formatted_history)
            return {"suggestedResponses": suggestions}

        # Otherwise return the last known responses or some default
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
async def get_suggestions(message: str = ""):
    """
    Simple endpoint that returns suggestions for a given message.
    """
    try:
        if not message:
            return {
                "suggestions": ["What's trending now?", "Style advice please", "Tell me about tabis"]
            }
        # Generate suggestions ignoring any conversation context
        suggestions = generate_suggestions(message, "")
        return {"suggestions": suggestions}
    except Exception as e:
        print("Error in /api/suggestions:", e)
        return {"suggestions": ["Try a different question", "Style advice please", "What's trending?"]}

if __name__ == "__main__":
    # Just run your FastAPI app (uvicorn, etc.)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)