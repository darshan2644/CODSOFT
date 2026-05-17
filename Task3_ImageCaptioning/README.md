# Task 3 - Image Captioning

Automatically generates a text description for any image using the **Salesforce BLIP** model. You give it an image (from a file or a URL), and it returns a caption describing what's in the image. Powered by a pre-trained deep learning model.

> **Note:** The first run will download ~900MB of model weights from Hugging Face. This is a one-time download.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `transformers` | Loads the BLIP image captioning model and processor from Hugging Face |
| `torch` | Deep learning backend that runs the model |
| `Pillow` | Opens and processes image files |
| `requests` | Downloads images from a URL if a link is provided |

---

## How to Run

**Step 1 — Install dependencies:**

```bash
pip install transformers torch Pillow requests
```

**Step 2 — Run the script:**

```bash
python image_captioning.py
```

---

## Sample Output

```
============================================================
        Image Captioning using BLIP Model
============================================================

Loading BLIP model... (first run downloads ~900MB)
Model loaded successfully!

Enter image path or URL (or 'quit' to exit): test.jpg

Processing image...

Caption: a dog sitting on a wooden floor next to a window

------------------------------------------------------------

Enter image path or URL (or 'quit' to exit): https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/600px-Cat03.jpg

Downloading image from URL...
Processing image...

Caption: a cat sitting on a white surface looking at the camera

------------------------------------------------------------

Enter image path or URL (or 'quit' to exit): quit

Thank you for using Image Captioning!
============================================================
```
