"""
Application runner script.
"""
import uvicorn
from src.config import settings

if __name__ == "__main__":
    print(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=True
    ) 