from google.genai import types

def save_message(role, text):
    return types.Content(
        role=role,
        parts=[types.Part.from_text(text=text)]
    )
