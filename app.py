import os
import re
import base64
import torch
import gradio as gr
from PIL import Image
from groq import Groq
from gtts import gTTS
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr

# ------------------------------------------------------------------
# ✅ CONFIGURATION
# ------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("⚠️ Missing GROQ_API_KEY environment variable.")

client = Groq(api_key=GROQ_API_KEY)
system_prompt = (
    "You are a professional pharmacist. Identify patient details (name, DOB), "
    "medicines, dosage, frequency, and doctor's name from the prescription."
)

# ------------------------------------------------------------------
# ✅ OCR MODELS (EasyOCR + TrOCR)
# ------------------------------------------------------------------
print("Loading EasyOCR (English + Hindi)...")
reader = easyocr.Reader(["en", "hi"], gpu=torch.cuda.is_available())

try:
    print("Loading TrOCR...")
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    if torch.cuda.is_available():
        trocr_model = trocr_model.to("cuda")
    print("TrOCR ready.")
except Exception as e:
    trocr_processor = None
    trocr_model = None
    print("⚠️ TrOCR failed to load:", e)


def run_trocr(image_path: str) -> str:
    """Run handwritten OCR using TrOCR."""
    if not trocr_processor or not trocr_model:
        return ""
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")
        generated_ids = trocr_model.generate(pixel_values)
        return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        return f"[TrOCR error: {e}]"


# ------------------------------------------------------------------
# ✅ OCR + INFO EXTRACTION
# ------------------------------------------------------------------
def extract_prescription_info(image_path: str):
    """Combine EasyOCR + TrOCR outputs, then extract fields."""
    easy_lines = reader.readtext(image_path, detail=0)
    easy_text = "\n".join(easy_lines)
    trocr_text = run_trocr(image_path)

    full_text = "\n".join(dict.fromkeys((easy_text + "\n" + trocr_text).splitlines()))

    info = {
        "Patient": None,
        "Doctor": None,
        "Medicines": [],
        "Dosage": [],
        "Frequency": [],
    }

    # Simple regex parsing
    m = re.search(r"(?:Patient|Name)[:\-\s]*(.+)", full_text, re.IGNORECASE)
    if m:
        info["Patient"] = m.group(1).strip()

    m = re.search(r"(?:Dr\.?|Doctor)[:\-\s]*(.+)", full_text, re.IGNORECASE)
    if m:
        info["Doctor"] = m.group(1).strip()

    meds = re.findall(r"([A-Z][A-Za-z0-9\-\s]{1,40}?)(?:\s+(\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg)))?", full_text)
    for name, dose in meds:
        if name.strip() and name.strip() not in info["Medicines"]:
            info["Medicines"].append(name.strip())
        if dose.strip() and dose.strip() not in info["Dosage"]:
            info["Dosage"].append(dose.strip())

    freq_matches = re.findall(
        r"\b(?:once daily|twice daily|thrice daily|once a day|twice a day|q\d+h|bid|tid|qid|od|prn|as needed)\b",
        full_text,
        re.IGNORECASE,
    )
    info["Frequency"] = list(dict.fromkeys([f.strip() for f in freq_matches]))

    return full_text, info


# ------------------------------------------------------------------
# ✅ HELPER FUNCTIONS
# ------------------------------------------------------------------
def analyze_with_groq(query: str, image_path: str) -> str:
    """Send OCR result to Groq LLM for structured interpretation."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ],
            }
        ],
    )
    return completion.choices[0].message.content


def text_to_speech(text: str, filename="response.mp3"):
    """Generate voice output."""
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        return filename
    except Exception as e:
        return f"[TTS error: {e}]"


# ------------------------------------------------------------------
# ✅ MAIN PROCESS FUNCTION
# ------------------------------------------------------------------
def process_inputs(audio_file, image_file, strip_size=10):
    if not image_file:
        return "No image provided.", "", "", "", None

    ocr_text, info = extract_prescription_info(image_file)
    query = system_prompt + "\n\nExtracted Text:\n" + ocr_text

    response = analyze_with_groq(query, image_file)
    reminders, restock = generate_reminders(response, strip_size)
    audio_path = text_to_speech(response)
    return ocr_text, response, reminders, restock, audio_path


def generate_reminders(text, strip_size):
    """Generate reminder & restock messages."""
    text = text.lower()
    freq = 0
    if "once daily" in text or "od" in text:
        freq = 1
    elif "twice daily" in text or "bid" in text:
        freq = 2
    elif "thrice daily" in text or "tid" in text:
        freq = 3
    elif "prn" in text or "as needed" in text:
        freq = 0

    if freq == 0:
        reminder = "Take medicine as needed (PRN)."
        restock = "Restock depends on usage."
    else:
        reminder = f"Take your medicine {freq} time(s) a day."
        restock = f"One strip ({strip_size} pills) lasts about {strip_size // freq} days."
    return reminder, restock


# ------------------------------------------------------------------
# ✅ GRADIO UI
# ------------------------------------------------------------------
css = """
#chatbot-box {height: 420px; overflow-y: auto; padding: 10px; background-color: #161b22;
              border-radius: 12px; border: 1px solid #30363d;}
.chat-message {display:flex; margin-bottom:12px;}
.chat-message.user {justify-content:flex-end;}
.chat-bubble {padding:10px 14px; border-radius:18px; max-width:70%; line-height:1.4;}
.chat-message.user .chat-bubble {background-color:#238636; color:white;}
.chat-message.bot .chat-bubble {background-color:#30363d; color:#e6edf3;}
"""

with gr.Blocks(css=css, title="PharmaBot") as demo:
    gr.Markdown("# PharmaBot — Multilingual OCR + AI Prescription Assistant")

    with gr.Tab("Prescription Analyzer"):
        with gr.Row():
            img_in = gr.Image(sources=["upload", "webcam"], type="filepath", label="Upload / Capture Prescription")
        strip_in = gr.Dropdown(choices=[8, 10, 14, 20, 28, 30], value=10, label="Pills per Strip")
        process_btn = gr.Button("Process", variant="primary")
        with gr.Accordion("Results", open=True):
            ocr_out = gr.Textbox(label="Extracted Text")
            ai_out = gr.Textbox(label="Pharmacist's Interpretation")
            reminder_out = gr.Textbox(label="Reminder")
            restock_out = gr.Textbox(label="Restock Info")
            audio_out = gr.Audio(label="Pharmacist Voice")
        process_btn.click(process_inputs, inputs=[None, img_in, strip_in],
                          outputs=[ocr_out, ai_out, reminder_out, restock_out, audio_out])

    gr.Markdown("*(c) 2025 PharmaBot — AI Prescription Chatbot*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 8080)))
