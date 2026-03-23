"""
Content processing module using GPT-4 for structuring.
"""

import os
import json
from typing import Optional, List
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv

load_dotenv()


class StructuredNote(BaseModel):
    """Structured note output."""
    title: str = Field(description="Concise title for the note")
    category: str = Field(description="Category: idea/todo/meeting/inspiration/memo")
    priority: str = Field(description="Priority: high/medium/low")
    summary: str = Field(description="Brief summary of the content")
    key_points: List[str] = Field(description="Key extracted points")
    tags: List[str] = Field(description="Relevant tags")
    sentiment: str = Field(description="Emotional tone: excited/hesitant/urgent/calm")
    action_items: List[str] = Field(description="Actionable items if any")


class Processor:
    """Process transcribed text into structured notes using GPT-4."""
    
    CATEGORIES = ["idea", "todo", "meeting", "inspiration", "memo"]
    PRIORITIES = ["high", "medium", "low"]
    SENTIMENTS = ["excited", "hesitant", "urgent", "calm", "neutral"]
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize processor.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
        """
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = model
    
    def process(self, text: str, context: Optional[str] = None) -> StructuredNote:
        """
        Process text into structured note.
        
        Args:
            text: Transcribed text
            context: Optional context about the note
            
        Returns:
            StructuredNote: Structured note object
        """
        system_prompt = """You are an expert note-taking assistant. Analyze the input text and extract structured information.

Categories:
- idea: Creative thoughts, innovations, concepts
- todo: Tasks, action items, things to do
- meeting: Meeting notes, discussions, decisions
- inspiration: Quotes, insights, motivational content
- memo: General notes, reminders, observations

Sentiment detection:
- excited: Enthusiastic, energetic, positive
- hesitant: Uncertain, tentative, doubtful
- urgent: Time-sensitive, pressing, important
- calm: Relaxed, steady, neutral

Output valid JSON matching the StructuredNote schema."""

        user_prompt = f"""Please structure the following voice note:

{text}

{f'Context: {context}' if context else ''}

Analyze and return structured JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        return StructuredNote(**data)
    
    def batch_process(self, texts: List[str]) -> List[StructuredNote]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of transcribed texts
            
        Returns:
            List of structured notes
        """
        return [self.process(text) for text in texts]
