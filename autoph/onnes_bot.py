import discord
import asyncio
from io import BytesIO
import fitz  # PyMuPDF
import os

DISCORD_TOKEN = os.environ.get("DISCORD_ONNES_TOKEN", None)
CHANNEL_ID = os.environ.get("DISCORD_ONNES_CHANNEL_ID", None)

# Function to convert PDF to PNG in memory using PyMuPDF
def convert_pdf_to_png_in_memory(pdf_file):
    pdf_document = fitz.open(pdf_file)
    # Get the only page
    page = pdf_document.load_page(0)
    # Convert the page to an image
    image = page.get_pixmap(alpha=True)
    # Convert the image to bytes
    img_bytes = image.tobytes("png")
    return img_bytes

# Function to send a message to a Discord channel
async def send_message(channel_id, message_content, image_paths=None):
    # Initialize the Discord client
    client = discord.Client()

    # Event handler for when the bot is ready
    @client.event
    async def on_ready():
        print(f'We have logged in as {client.user}')

        # Get the channel object using the provided channel ID
        channel = (client.get_channel(channel_id) or await client.fetch_channel(channel_id))

        # Send the message content
        await channel.send(content=message_content)

        # Send images if provided
        if image_paths:
            for image_path in image_paths:
                # Convert PDF to PNG in memory using PyMuPDF
                png_image = convert_pdf_to_png_in_memory(image_path)
                # Create a discord.File object from the PNG image bytes
                png_file = discord.File(BytesIO(png_image), filename=image_path.replace('pdf', 'png'))
                # Send the PNG image
                await channel.send(file=png_file)

        # Close the Discord client
        await client.close()

    # Run the client with the provided bot token
    await client.start(DISCORD_TOKEN)

# Standalone function to send Discord message with images
def send_discord_message_with_images():
    # Check if Tc.dat file exists
    if os.path.isfile("Tc.dat"):
        with open("Tc.dat") as f:
            la, wlog, Tc_AD, Tc_OPT = [float(i.split()[-1].split("±")[0]) for i in f.readlines()]

        mat_id = os.path.basename(os.getcwd())
        # Message content
        message_content = f"{mat_id} -> Tc = {Tc_AD:.1f}, lambda = {la:.1f}, wlog = {wlog:.1f}"
        # Run the script asynchronously
        asyncio.run(send_message(CHANNEL_ID, message_content, ["el-bs.pdf", "ph-bs.pdf"]))

if __name__ == "__main__":
    send_discord_message_with_images()
