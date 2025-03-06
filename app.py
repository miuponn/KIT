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
    "suggestedResponses": ["What should I wear to a gallery opening?", "Suggest styling tips based on my wardrobe.", "Break down a trend: tabis"]
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
        # detect if we're in outfit building mode and at specific stages
        if "what's the occasion for this outfit" in bot_reply.lower():
            return ["Wedding", "Job interview", "Casual weekend"]
            
        if any(q in bot_reply.lower() for q in ["what's the vibe", "colour tones", "silhouettes", "weather", "budget"]):
            # generate contextual answers to the specific question
            suggestion_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate 3 brief user answers (not questions) to respond to the assistant's question. "
                            "These should be plausible answers a user might give to the fashion question just asked. "
                            "Format as a simple comma-separated list without numbering or quotes."
                        )
                    },
                    {"role": "assistant", "content": bot_reply},
                ],
                max_tokens=50
            )
            suggestions_text = suggestion_response.choices[0].message.content
            suggestions = [s.strip() for s in suggestions_text.split(",") if s.strip()]
            return suggestions[:3]
            
        # after outfit generation (detect outfit formatting)
        if "## " in bot_reply and any(item in bot_reply for item in ["TOP:", "BOTTOM:", "FOOTWEAR:"]):
            return ["Shopping links please", "I'd like to swap some items", "This looks perfect"]
            
        # after shopping links
        if "here are available links" in bot_reply.lower():
            return ["Save this outfit for me", "I'd like to swap some items", "Add final touches"]
            
        # when asked what to change
        if "what would you like to change" in bot_reply.lower():
            return ["Different shoes", "Something more affordable", "Different color palette"]

        # default case - generate standard suggestions
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
            max_tokens=250,
        )
        bot_reply = response.choices[0].message.content
        # generate follow-up suggestions
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
async def get_suggestions(message: str = ""):
    """
    Simple endpoint that returns suggestions for a given message.
    """
    try:
        if not message:
            return {
                "suggestions": ["What's trending now?", "Style advice please", "Break down a trend: tabis"]
            }
        # generate suggestions ignoring any conversation context
        suggestions = generate_suggestions(message, "")
        return {"suggestions": suggestions}
    except Exception as e:
        print("Error in /api/suggestions:", e)
        return {"suggestions": ["Try a different question", "Style advice please", "What's trending?"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)