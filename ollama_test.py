
from ollama import chat

response = chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "How do I troubleshoot a VPN connection failure?"
        }
    ]
)

print("AI Response:")
print(response.message.content)
