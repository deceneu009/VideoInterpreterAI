from app.scripts import generateResponse
from PIL import Image

if __name__ == "__main__":
    with Image.open("hand.jpeg") as im:
        generateResponse("How many fingers are there?", im)
