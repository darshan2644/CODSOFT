"""
Image Captioning AI using BLIP (Bootstrapped Language-Image Pretraining)
========================================================================
Author  : Antigravity AI
Project : CODSOFT Internship - Task 3

INSTALLATION (run once in your terminal before using this script):
    pip install transformers torch Pillow requests

WHAT IS BLIP?
    BLIP (Bootstrapped Language-Image Pretraining) is a vision-language
    model developed by Salesforce Research (2022). It can understand
    both images and text at the same time.

    Architecture overview:
    ┌─────────────┐        ┌──────────────────────┐
    │  Image      │──────▶ │  Vision Encoder      │
    │  (pixels)   │        │  (ViT – Vision       │
    └─────────────┘        │   Transformer)       │
                           └──────────┬───────────┘
                                      │  visual features
                           ┌──────────▼───────────┐
                           │  Language Model       │
                           │  (BERT-style decoder) │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │  Caption text output  │
                           └──────────────────────┘

HOW VISION-LANGUAGE MODELS WORK (simplified):
    1. The Vision Encoder (ViT) splits the image into small patches
       and encodes each patch into a high-dimensional vector.
    2. These visual feature vectors are fed into a Language Model
       alongside a text prompt ("a photography of").
    3. The Language Model uses cross-attention to "look at" the image
       features while predicting the next word, one token at a time.
    4. The result is a natural-language caption describing the image.

NOTE: First run will download ~900 MB of model weights from Hugging Face.
      Subsequent runs use the cached weights and start instantly.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import os
import sys

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MODEL_ID   = "Salesforce/blip-image-captioning-base"
DEMO_URL   = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/"
    "280px-PNG_transparency_demonstration_1.png"
)


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_model():
    """
    Download (first run) or load (cached) the BLIP processor and model.
    Auto-detects GPU (CUDA) or falls back to CPU.

    Returns
    -------
    processor : BlipProcessor  – handles image/text pre-processing
    model     : BlipForConditionalGeneration – the neural network
    device    : str – "cuda" or "cpu"
    """
    print("\n  ⏳  Loading BLIP model …")
    print(f"  Model : {MODEL_ID}")
    print("  ⚠   First run downloads ~900 MB. Please be patient.\n")

    # Detect available hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  🖥   Device : {device.upper()}")
    if device == "cpu":
        print("  ℹ   No GPU detected — running on CPU (slower but works fine).")

    # Load processor (tokenizer + image feature extractor bundled together)
    processor = BlipProcessor.from_pretrained(MODEL_ID)

    # Load the model and move it to the selected device
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()          # switch to inference mode (disables dropout etc.)

    print("  ✅  Model loaded successfully!\n")
    return processor, model, device


# ---------------------------------------------------------------------------
# CAPTION GENERATION
# ---------------------------------------------------------------------------

def generate_caption(image, processor, model, device):
    """
    Generate a natural-language caption for a PIL Image.

    Parameters
    ----------
    image     : PIL.Image – the image to describe
    processor : BlipProcessor
    model     : BlipForConditionalGeneration
    device    : str

    Returns
    -------
    str – the generated caption
    """
    # BLIP expects RGB images; convert in case it's RGBA / grayscale / etc.
    image = image.convert("RGB")

    # The processor turns the raw PIL image into tensors the model understands.
    # return_tensors="pt" gives us PyTorch tensors.
    inputs = processor(images=image, return_tensors="pt")

    # Move all input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # model.generate() runs Beam Search:
    #   num_beams=5  → keep 5 candidate sequences at each step
    #   max_length   → hard cap on output length
    #   early_stopping → stop all beams once EOS token is generated
    # torch.no_grad() disables gradient tracking to save memory & speed up.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length    = 50,
            num_beams     = 5,
            early_stopping= True,
        )

    # Decode the token IDs back into human-readable text.
    # skip_special_tokens=True removes [CLS], [SEP], [PAD] etc.
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption


# ---------------------------------------------------------------------------
# IMAGE LOADING — FROM URL
# ---------------------------------------------------------------------------

def caption_from_url(url, processor, model, device):
    """
    Fetch an image from the internet by URL and generate a caption.

    Returns the caption string, or None on failure.
    """
    print(f"\n  🌐  Fetching image from URL …\n  {url}")
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()                # raise error on 4xx/5xx
        image = Image.open(response.raw)
        print("  ✅  Image fetched successfully.")
        return generate_caption(image, processor, model, device)

    except requests.exceptions.ConnectionError:
        print("  ❌  Connection error – check your internet connection.")
    except requests.exceptions.Timeout:
        print("  ❌  Request timed out – the server took too long to respond.")
    except requests.exceptions.HTTPError as e:
        print(f"  ❌  HTTP error : {e}")
    except Exception as e:
        print(f"  ❌  Could not open image from URL : {e}")

    return None


# ---------------------------------------------------------------------------
# IMAGE LOADING — FROM LOCAL FILE
# ---------------------------------------------------------------------------

def caption_from_file(filepath, processor, model, device):
    """
    Open a local image file and generate a caption.

    Returns the caption string, or None on failure.
    """
    filepath = filepath.strip().strip('"').strip("'")   # clean pasted paths

    if not os.path.exists(filepath):
        print(f"  ❌  File not found : {filepath}")
        print("  ℹ   Double-check the path and try again.")
        return None

    print(f"\n  📂  Opening local file : {filepath}")
    try:
        image = Image.open(filepath)
        print("  ✅  Image opened successfully.")
        return generate_caption(image, processor, model, device)

    except Exception as e:
        print(f"  ❌  Could not open image file : {e}")

    return None


# ---------------------------------------------------------------------------
# BANNER & MENU
# ---------------------------------------------------------------------------

def print_banner():
    """Print the application header."""
    print("\n" + "=" * 50)
    print("   🖼   IMAGE CAPTIONING AI")
    print("        CodSoft Task 3  —  BLIP Model")
    print("=" * 50)
    print("  Powered by : Salesforce/blip-image-captioning-base")
    print("  Describe any image with one line of text!\n")


def print_menu():
    """Display the main option menu."""
    print("  ┌──────────────────────────────┐")
    print("  │  Choose an option :          │")
    print("  │  1. Caption from image URL   │")
    print("  │  2. Caption from local file  │")
    print("  │  3. Try sample demo image    │")
    print("  │  4. Exit                     │")
    print("  └──────────────────────────────┘")


# ---------------------------------------------------------------------------
# MAIN APPLICATION LOOP
# ---------------------------------------------------------------------------

def main():
    print_banner()

    # Load the model once at startup; reuse across all captions
    try:
        processor, model, device = load_model()
    except Exception as e:
        print(f"\n  ❌  Failed to load model : {e}")
        print("  ℹ   Make sure you have run:")
        print("      pip install transformers torch Pillow requests")
        sys.exit(1)

    while True:
        print_menu()

        choice = input("\n  Enter option (1-4): ").strip()

        caption = None       # reset for each iteration

        # ── Option 1: URL ──────────────────────────────────────────────────
        if choice == "1":
            url = input("  Paste image URL : ").strip()
            if url:
                caption = caption_from_url(url, processor, model, device)
            else:
                print("  ⚠   No URL entered.")

        # ── Option 2: Local file ───────────────────────────────────────────
        elif choice == "2":
            filepath = input("  Enter path to image file : ").strip()
            if filepath:
                caption = caption_from_file(filepath, processor, model, device)
            else:
                print("  ⚠   No file path entered.")

        # ── Option 3: Demo image ───────────────────────────────────────────
        elif choice == "3":
            print("\n  🎯  Running demo with sample image …")
            caption = caption_from_url(DEMO_URL, processor, model, device)

        # ── Option 4: Exit ─────────────────────────────────────────────────
        elif choice == "4":
            print("\n  👋  Goodbye! Thanks for using Image Captioning AI.\n")
            break

        else:
            print("  ⚠   Invalid option. Please enter 1, 2, 3, or 4.")

        # ── Display caption if one was generated ───────────────────────────
        if caption:
            print("\n" + "─" * 50)
            print(f"  🖼   Generated Caption :")
            print(f"       \"{caption}\"")
            print("─" * 50)

        # ── Ask to continue or exit ────────────────────────────────────────
        if choice in ("1", "2", "3"):
            again = input("\n  Caption another image? (yes/no) : ").strip().lower()
            if again not in ("yes", "y"):
                print("\n  👋  Goodbye! Thanks for using Image Captioning AI.\n")
                break

        print()    # blank line before next menu


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
