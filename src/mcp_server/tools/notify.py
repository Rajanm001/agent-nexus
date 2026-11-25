"""Email notification tool for completed reports."""

class NotificationTool:
    async def send(self, to: str, subject: str, body: str) -> dict:
        # Production: sends via SMTP
        return {"sent": True, "to": to}
