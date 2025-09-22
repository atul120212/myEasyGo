import time
import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import os, re, json, logging
from serpapi import GoogleSearch
import random
import requests
from datetime import datetime

# --- Environment Variable Setup ---
load_dotenv()

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except KeyError:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables. Please create a .env file and add it.")

# --- Pydantic Models for Data Validation ---
class TripDetails(BaseModel):
    source: str
    destination: str
    startDate: str
    endDate: str
    duration: int
    travelers: int
    interests: List[str]
    budget: int

class Activity(BaseModel):
    type: str
    time: str
    title: str
    description: str
    image: str

class DayPlan(BaseModel):
    day: int
    title: str
    summary: str
    activities: List[Activity]

class Itinerary(BaseModel):
    title: str
    days: List[DayPlan]
    totalCost: int
    image_url: str | None = None

# --- NEW: Flight Models ---
class FlightInfo(BaseModel):
    airline: str
    airplane: str
    departure_time: str
    arrival_time: str
    departure_airport: str
    arrival_airport: str
    duration: str
    price: str
    travel_class: str = "Economy"

class FlightResponse(BaseModel):
    flights: List[FlightInfo]

# --- NEW: Hotel Models ---
class HotelInfo(BaseModel):
    name: str
    rating: Optional[str] = "4.0"
    description: str
    rate: str
    amenities: Optional[List[str]] = []
    location: Optional[str] = ""

class HotelResponse(BaseModel):
    hotels: List[HotelInfo]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="TripsAI API",
    description="API for generating personalized travel itineraries using Google Gemini.",
    version="2.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI Prompt Engineering Function ---
def create_gemini_prompt(details: TripDetails) -> str:
    interests_str = ", ".join(details.interests)
    
    json_format_instructions = """
    {
      "title": "A string for the itinerary title",
      "days": [
        {
          "day": "An integer for the day number",
          "title": "A string for the day's theme or title",
          "summary": "A short string summary of the day",
          "activities": [
            {
              "type": "A string for the activity type (e.g., 'foodie', 'adventure')",
              "time": "A string for the time of day (e.g., 'Morning', 'Afternoon', 'Evening')",
              "title": "A string for the activity title",
              "description": "A string describing the activity",
              "image": "A string URL for a placeholder image from 'https://placehold.co/100x100/..."
            }
          ]
        }
      ],
      "totalCost": "An integer representing the total estimated cost in INR"
    }
    """

    prompt = f"""
    You are an expert travel planner for India. Your task is to create a personalized travel itinerary based on the user's preferences.
    
    **User Preferences:**
    - source: {details.source}
    - Destination: {details.destination}
    - Duration: {details.duration} days
    - Number of Travelers: {details.travelers}
    - Budget (per person): INR {details.budget}
    - Interests: {interests_str}

    **Your Task:**
    1.  Generate a creative, logical, and exciting day-by-day itinerary.
    2.  The `totalCost` should be a realistic estimate in INR for the specified number of travelers, considering the budget level.
    3.  For each `activity` in the `itinerary`, the `description` should be detailed. Include specific, realistic (but fictional) details like:
        - **Famous food places:** Suggest a well-known local eatery and a famous dish to try.
        - **Visiting hours:** Mention typical opening and closing times for attractions (e.g., "open from 9 AM to 5 PM").
        - **Hotels and Flights:** Suggest realistic hotel names and approximate prices per night, and mention flight details (e.g., "A morning flight with IndiGo").
    4.  For each activity, provide a relevant placeholder image URL from `https://placehold.co/`. For example: `https://placehold.co/100x100/3498db/ffffff?text=Beach`.
    5.  The final output MUST be a single, valid JSON object that strictly follows this structure. Do not include any text, explanations, or markdown formatting before or after the JSON object.

    **Required JSON Structure:**
    {json_format_instructions}
    """
    return prompt

@app.post("/api/generate-itinerary", response_model=Itinerary)
async def generate_itinerary_endpoint(details: TripDetails):
    print("Received request with details:", details.model_dump_json(indent=2))
    
    prompt = create_gemini_prompt(details)
    
    try:
        response = model.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        itinerary_data = json.loads(cleaned_response_text)

        # Add image_url if missing
        if "image_url" not in itinerary_data or not itinerary_data["image_url"]:
            try:
                if os.environ.get("SERPAPI_KEY"):
                    params = {
                        "engine": "google_images",
                        "q": details.destination,
                        "api_key": os.environ.get("SERPAPI_KEY")
                    }
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    images_results = results.get("images_results", [])

                    if images_results:
                        valid_images = [
                            img for img in images_results
                            if "original" in img and "gstatic" not in img["original"]
                        ]

                        if valid_images:
                            best_image = max(
                                valid_images,
                                key=lambda img: img.get("original_width", 0) * img.get("original_height", 0)
                            )
                            image_url = best_image["original"]
                        else:
                            encoded_destination = details.destination.replace(" ", "+")
                            image_url = f"https://source.unsplash.com/1200x600/?{encoded_destination},travel"
                    else:
                        encoded_destination = details.destination.replace(" ", "+")
                        image_url = f"https://source.unsplash.com/1200x600/?{encoded_destination},travel"
                else:
                    encoded_destination = details.destination.replace(" ", "+")
                    image_url = f"https://source.unsplash.com/1200x600/?{encoded_destination},travel"

                itinerary_data["image_url"] = image_url
                
            except Exception as e:
                print(f"Failed to fetch image: {e}")
                encoded_destination = details.destination.replace(" ", "+")
                itinerary_data["image_url"] = f"https://source.unsplash.com/1200x600/?{encoded_destination},travel"

        itinerary = Itinerary(**itinerary_data)
        return itinerary

    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error processing AI response: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to process the itinerary from the AI. The response was not in the expected format."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred while generating the itinerary: {str(e)}"
        )

@app.get("/api/flights", response_model=FlightResponse)
async def get_flights(
    source_city: str = Query(..., description="Source city, e.g. Mumbai"),
    destination_city: str = Query(..., description="Destination city, e.g. Goa"),
    travel_date: str = Query(None, description="Travel date YYYY-MM-DD"),
):
    """
    Generate flight data using Gemini API.
    """
    try:
        logger.info(f"Received GET request for /api/flights with source_city='{source_city}', destination_city='{destination_city}', travel_date='{travel_date}'")
        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        Generate a JSON list of exactly 3 realistic flight options from {source_city} to {destination_city}.
        Travel date: {travel_date or "any upcoming date"}.

        Each flight must have these EXACT fields:
        - airline: string (e.g., "IndiGo", "Air India", "Vistara")
        - airplane: string (e.g., "Airbus A320", "Boeing 737")
        - departure_time: string in 12-hour format (e.g., "08:30 AM")
        - arrival_time: string in 12-hour format (e.g., "10:45 AM")
        - departure_airport: string with airport name and code (e.g., "Chhatrapati Shivaji International (BOM)")
        - arrival_airport: string with airport name and code (e.g., "Goa International (GOI)")
        - duration: string (e.g., "2h 15m")
        - price: string in INR format (e.g., "₹4,500")
        - travel_class: string (default "Economy")

        Return ONLY valid JSON with this structure:
        {{
          "flights": [
            {{ flight object }},
            {{ flight object }},
            {{ flight object }}
          ]
        }}

        Do not add any text before or after the JSON.
        """

        response = model.generate_content(prompt)
        text_output = response.text.strip()
        logger.info(f"Raw Gemini flights output: {text_output[:500]}...")

        # Extract JSON using regex
        match = re.search(r"\{[\s\S]*\}", text_output)
        if not match:
            logger.error("Regex failed to find JSON object.")
            raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")

        json_str = match.group(0)
        logger.info(f"Extracted JSON string: {json_str[:500]}...")
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        try:
            flights_data = json.loads(json_str)
            logger.info("JSON successfully parsed into Python dictionary.")
        except json.JSONDecodeError as e:
            logger.error("JSON parsing error: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {e}")

        # Validate the response
        try:
            validated_response = FlightResponse(**flights_data)
            logger.info("Pydantic validation successful.")
            return validated_response
        except ValidationError as e:
            logger.error("Pydantic validation failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Pydantic validation failed: {e}")

    except Exception as e:
        logger.error("Error generating flights: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate flights: {str(e)}")



@app.get("/api/hotels", response_model=HotelResponse)
async def get_hotels(query: str = Query(..., description="Hotel search query, e.g. 'Goa hotels'")):
    """
    Generate hotel data using Gemini API.
    """
    try:
        logger.info(f"Received GET request for /api/hotels with query='{query}'")
        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        Generate a JSON list of exactly 3 realistic hotel options for the query: "{query}".

        Each hotel must have these EXACT fields:
        - name: string (realistic hotel name)
        - rating: string (e.g., "4.2", "4.5")
        - description: string (brief description of the hotel)
        - rate: string in INR format (e.g., "₹3,500")
        - amenities: list of strings (e.g., ["WiFi", "Pool", "Spa"])
        - location: string (area/location description)

        Return ONLY valid JSON with this structure:
        {{
          "hotels": [
            {{ hotel object }},
            {{ hotel object }},
            {{ hotel object }}
          ]
        }}

        Do not add any text before or after the JSON.
        """

        response = model.generate_content(prompt)
        text_output = response.text.strip()
        logger.info(f"Raw Gemini hotels output: {text_output[:500]}...")

        # Extract JSON using regex
        match = re.search(r"\{[\s\S]*\}", text_output)
        if not match:
            logger.error("Regex failed to find JSON object.")
            raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")

        json_str = match.group(0)
        logger.info(f"Extracted JSON string: {json_str[:500]}...")
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        try:
            hotels_data = json.loads(json_str)
            logger.info("JSON successfully parsed into Python dictionary.")
        except json.JSONDecodeError as e:
            logger.error("JSON parsing error: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {e}")

        # Validate the response
        try:
            validated_response = HotelResponse(**hotels_data)
            logger.info("Pydantic validation successful.")
            return validated_response
        except ValidationError as e:
            logger.error("Pydantic validation failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Pydantic validation failed: {e}")

    except Exception as e:
        logger.error("Error generating hotels: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate hotels: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the TripsAI API. Visit /docs for documentation."}