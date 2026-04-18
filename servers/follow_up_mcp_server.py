import os
from fastmcp import FastMCP
from anthropic import Anthropic

follow_up_mcp = FastMCP(name="follow-up-mcp")

SYSTEM_PROMPT = (
    "You are a professional care provider guide. Your role is to help users "
    "navigate care-related topics by asking targeted follow-up questions. "
    "Based on the conversation so far, generate 1-2 follow-up questions that "
    "are clarifying, instructional, or aimed at understanding the user's "
    "situation better. Your tone should be warm but professional and on-point. "
    "Respond in a natural conversational style — do not use bullet lists or "
    "numbered lists. Keep your response concise."
)


@follow_up_mcp.tool(
    name="generate_follow_up_questions",
    description=(
        "Use this tool when the user's input is too vague, too general, ambiguous, "
        "overly broad, or too complex to respond to directly. Call this tool to get "
        "1-2 targeted follow-up questions before attempting an answer. The questions "
        "help clarify the user's intent, understand their situation better, or guide "
        "them with instructional prompts. Input the last few conversation turns as a "
        "list of strings (e.g. ['user: I need help with care', 'assistant: Sure, "
        "what kind of care?', 'user: just general care stuff'])."
    ),
    tags={"follow-up", "clarification"},
)
async def generate_follow_up_questions(chat_turns: list[str]) -> str:
    """Generate 1-2 follow-up questions based on recent chat history."""
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_content = "Here is the recent conversation:\n\n" + "\n".join(chat_turns)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    return response.content[0].text


if __name__ == "__main__":
    follow_up_mcp.run()
