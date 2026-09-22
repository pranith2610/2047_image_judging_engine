"""
Startup launcher for THE 2047 judging engine.
Usage: python run.py
"""
import uvicorn
from app.config import HOST, PORT

if __name__ == "__main__":
    print(f"[*] Starting THE 2047 - AI Image Comparison & Judging Engine on http://{HOST}:{PORT}")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
