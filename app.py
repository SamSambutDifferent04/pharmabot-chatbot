# Install dependencies (may take a few minutes). If Colab already has GPU-enabled torch, the torch install may be unnecessary.
!pip install --upgrade --quiet gradio easyocr transformers groq gTTS ffmpeg-python pydub SpeechRecognition langdetect
# Try installing a compatible torch; if this errors out, remove or comment the line and rely on Colab's preinstalled torch.
!pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || true
!apt-get update -qq && apt-get install -y -qq portaudio19-dev ffmpeg
print('Install step finished. Restart runtime if you hit CUDA/torch conflicts.')

# Imports & basic checks
import os, re, base64, logging
from google.colab import userdata
import gradio as gr
from groq import Groq
from gtts import gTTS
import torch
from PIL import Image
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pydub import AudioSegment
from io import BytesIO
from base64 import b64decode
from google.colab import output
from google.colab.output import eval_js
from IPython.display import Javascript, display
import speech_recognition as sr

logging.basicConfig(level=logging.INFO)
print('torch.cuda.is_available() =', torch.cuda.is_available())

# Initialize EasyOCR (English + Hindi)
print('Initializing EasyOCR (en + hi)...')
reader = easyocr.Reader(['en', 'hi'], gpu=torch.cuda.is_available())
print('EasyOCR ready.')

# TrOCR (large). Try to load, fallback cleanly if not possible.
try:
    print('Loading TrOCR (may take time)...')
    trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    if torch.cuda.is_available():
        trocr_model = trocr_model.to('cuda')
    print('TrOCR loaded.')
except Exception as e:
    trocr_processor = None
    trocr_model = None
    print('TrOCR load failed (will fallback to EasyOCR). Error:', e)

def run_trocr(image_path):
    if trocr_processor is None or trocr_model is None:
        return ""
    try:
        image = Image.open(image_path).convert('RGB')
        pixel_values = trocr_processor(images=image, return_tensors='pt').pixel_values
        if torch.cuda.is_available():
            pixel_values = pixel_values.to('cuda')
        generated_ids = trocr_model.generate(pixel_values)
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    except Exception as e:
        return f"[TrOCR error: {e}]"

  # Groq client using Colab userdata (keeps your previous workflow)
GROQ_KEY = userdata.get('medical_application_projects')
if not GROQ_KEY:
    print('Warning: Groq API key not found in userdata.get(\"medical_application_projects\").')
client = Groq(api_key=GROQ_KEY)
system_prompt = '''You are a professional pharmacist. Identify patient details (name, DOB), medicines, dosage, frequency, and doctor's name from the prescription.'''

def extract_prescription_info(image_path):
    '''Return (full_text, info_dict). Uses EasyOCR + TrOCR (if available).'''
    try:
        easy_lines = reader.readtext(image_path, detail=0) or []
        easy_text = "\n".join([l.strip() for l in easy_lines if l.strip()])
        trocr_text = run_trocr(image_path)
        # merge without duplicates
        merged = []
        for line in (easy_text + "\n" + trocr_text).splitlines():
            s = line.strip()
            if s and s not in merged:
                merged.append(s)
        full_text = "\n".join(merged)

        info = {'Patient': None, 'Doctor': None, 'Hospital': None, 'Medicines': [], 'Dosage': [], 'Frequency': []}

        m = re.search(r'(?:Patient|Name)[:\-\s]*(.+)', full_text, re.IGNORECASE)
        if m: info['Patient'] = m.group(1).strip()
        m = re.search(r'(?:Dr\.?|Doctor)[:\-\s]*(.+)', full_text, re.IGNORECASE)
        if m: info['Doctor'] = m.group(1).strip()
        m = re.search(r'(?:Hospital|Clinic)[:\-\s]*(.+)', full_text, re.IGNORECASE)
        if m: info['Hospital'] = m.group(1).strip()

        med_matches = re.findall(r'([A-Z][A-Za-z0-9\-\s]{1,40}?)(?:\s+(\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg)))?', full_text)
        for nm, dose in med_matches:
            nm = nm.strip()
            if nm and nm not in info['Medicines']:
                info['Medicines'].append(nm)
            if dose and dose.strip() not in info['Dosage']:
                info['Dosage'].append(dose.strip())

        freq_matches = re.findall(r'\b(?:once daily|twice daily|thrice daily|once a day|twice a day|q\d+h|bid|tid|qid|od|prn|as needed)\b', full_text, re.IGNORECASE)
        info['Frequency'] = list(dict.fromkeys([f.strip() for f in freq_matches]))

        return full_text, info
    except Exception as e:
        return f"[ERROR during extraction: {e}]", {'Patient': None, 'Doctor': None, 'Hospital': None, 'Medicines': [], 'Dosage': [], 'Frequency': []}

  def clean_text(text):
    if not text:
        return text
    text = re.sub(r'[*#]+', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def real_transcribe_with_groq(audio_filepath, stt_model='whisper-large-v3'):
    if not audio_filepath or not os.path.exists(audio_filepath):
        return '[INFO: No audio provided]'
    try:
        with open(audio_filepath, 'rb') as f:
            transcription = client.audio.transcriptions.create(file=f, model=stt_model, response_format='text')
        return getattr(transcription, 'text', transcription)
    except Exception as e:
        return f'[ERROR: STT failed → {e}]'

def real_analyze_image_with_query(query, image_filepath, model='meta-llama/llama-4-scout-17b-16e-instruct'):
    if not image_filepath or not os.path.exists(image_filepath):
        return '[ERROR: Image file not found]'
    try:
        with open(image_filepath, 'rb') as imgf:
            encoded = base64.b64encode(imgf.read()).decode('utf-8')
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role':'user','content':[{'type':'text','text':query},{'type':'image_url','image_url':{'url':f'data:image/png;base64,{encoded}'}}]}]
        )
        raw = completion.choices[0].message.content
        return clean_text(raw)
    except Exception as e:
        return f'[ERROR: Image analysis failed → {e}]'

def text_to_speech_with_gtts(text, outfile='final.mp3'):
    try:
        tts = gTTS(text=text or ' ', lang='en')
        tts.save(outfile)
        return outfile
    except Exception as e:
        return f'[ERROR: TTS failed → {e}]'

  def generate_reminders(doctor_text, strip_size=10):
    text = (doctor_text or '').lower()
    reminders = []
    restocks = []
    freq = None
    if re.search(r'\bonce\b.*\bdaily\b|\bod\b', text):
        reminders.append('Take your medicine once every 24 hours.')
        freq = 1
    elif re.search(r'\btwice\b.*\bdaily\b|\bbid\b', text):
        reminders.append('Take your medicine every 12 hours.')
        freq = 2
    elif 'prn' in text or 'as needed' in text:
        reminders.append('Take as needed (PRN).')
        freq = 0
    else:
        reminders.append('Frequency not clearly detected.')

    if freq and freq > 0:
        restocks.append(f'1 strip ({strip_size} pills) lasts ~{strip_size // freq} days.')
    else:
        restocks.append('Restock depends on usage.')

    return '\\n'.join(reminders), '\\n'.join(restocks)

def process_inputs(audio_filepath, image_filepath, strip_size=10):
    stt_text = real_transcribe_with_groq(audio_filepath) if audio_filepath else '[INFO: No audio]'
    if image_filepath:
        ocr_text, info = extract_prescription_info(image_filepath)
        query = system_prompt + '\\n\\nExtracted text:\\n' + ocr_text + '\\n\\nPatient voice input:\\n' + str(stt_text)
        doc_response = real_analyze_image_with_query(query, image_filepath)
        # update small memory
        if isinstance(info, dict):
            for k in ['Patient','Doctor','Hospital','Medicines','Dosage','Frequency']:
                if k in info:
                    prescription_memory[k] = info[k]
    else:
        doc_response = '[INFO: No image provided]'
    reminders, restock = generate_reminders(doc_response, strip_size)
    tts_path = None
    try:
        tts_path = text_to_speech_with_gtts(doc_response, 'final.mp3')
    except Exception:
        tts_path = None
    return stt_text, (doc_response or ''), reminders, restock, tts_path

prescription_memory = {'Patient': None, 'Doctor': None, 'Hospital': None, 'Medicines': [], 'Dosage': [], 'Frequency': []}

def chatbot_response(user_input):
    ui = (user_input or '').lower()
    if 'medicine' in ui or 'prescribed' in ui:
        return 'You have been prescribed: ' + (', '.join(prescription_memory.get('Medicines') or []) or 'No medicines detected.')
    if 'dosage' in ui:
        return 'Dosage: ' + (', '.join(prescription_memory.get('Dosage') or []) or 'Not detected.')
    if 'frequency' in ui or 'reminder' in ui:
        return 'Frequency: ' + (', '.join(prescription_memory.get('Frequency') or []) or 'Not detected.')
    if 'doctor' in ui:
        return 'Doctor: ' + (prescription_memory.get('Doctor') or 'Not detected.')
    if 'patient' in ui:
        return 'Patient: ' + (prescription_memory.get('Patient') or 'Not detected.')
    return "Hi — I'm PharmaBot. Upload a prescription or ask about medicines, dosage, or reminders!"

chat_history = []
def update_chat(history_html, user_msg):
    if not user_msg:
        return history_html or "<div id='chatbot-box'></div>",
    chat_history.append(('user', user_msg))
    reply = chatbot_response(user_msg)
    chat_history.append(('bot', reply))
    inner = ''.join([f"<div class='chat-message {s}'><div class='chat-bubble'>{m}</div></div>" for s, m in chat_history])
    html = f"<div id='chatbot-box'>" + inner + "</div>"
    return html, ""

from base64 import b64decode
def capture_from_webcam(filename='captured_prescription.jpg', quality=0.8):
    js = Javascript("""
    async function captureImage(quality) {
      const div = document.createElement('div');
      const video = document.createElement('video');
      const button = document.createElement('button');
      button.textContent = 'Capture Photo';
      div.appendChild(video);
      div.appendChild(button);
      document.body.appendChild(div);
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      video.srcObject = stream;
      await video.play();
      await new Promise(resolve => button.onclick = resolve);
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getTracks().forEach(t => t.stop());
      const data = canvas.toDataURL('image/jpeg', quality);
      div.remove();
      return data;
    }
    (async () => { window._captured = await captureImage(0.8); })();
    """)
    display(js)
    data = eval_js('window._captured')
    if not data:
        raise RuntimeError('No image captured')
    img_bytes = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(img_bytes)
    return filename

css = """
body {background-color: #0d1117; color: #e6edf3;}
.gradio-container {max-width: 100% !important; margin: 0 auto !important}
#chatbot-box {height: 420px; overflow-y: auto; padding: 10px; background-color: #161b22; border-radius: 12px; border: 1px solid #30363d; display:flex; flex-direction:column}
.chat-message {display:flex; margin-bottom:12px}
.chat-message.user {justify-content:flex-end}
.chat-bubble {padding:10px 14px; border-radius:18px; max-width:70%; line-height:1.4; word-wrap:break-word}
.chat-message.user .chat-bubble {background-color:#238636; color:white; border-bottom-right-radius:4px}
.chat-message.bot .chat-bubble {background-color:#30363d; color:#e6edf3; border-bottom-left-radius:4px}
.gr-button-primary {background-color:#238636 !important; color:white !important; border-radius:8px !important}
"""

faq_dict = {
    "What are your pharmacy’s working hours?": " Our pharmacy works 24/7 for the user's convinience",
    "Do you offer home delivery for medicines?": "yes we offer home delivery but pharmacies may charge extra depending on the distance",
    "How can I upload my prescription?": "Go to the user dashboard page. There under 'Prescriptions' section, there will be an option for uploading a medical prescription",
    "Can I order medicines without a prescription?": "Yes you can order medicines without a prescription but only those of general use",
    "Do you have cash on delivery option?": "Yes we have a cash on delivery option but it has a higher charge compared to online payment",
    "Can I track my medicine order?": "Yes you can track your medicine order thro",
    "How do I cancel or modify an order?": "Go to your user dashboard page. There you will find an option to cancel/modify your medicine order",
    "What payment methods do you accept?": "We accept cash on delivery, card as well as online payment",
    "Do you have discounts or offers available?": "At the moment we currently do not have discounts or offers available",
    "Can I return or exchange medicines?": "Yes you can return or exchange medicine but the medicine you ordered will be examined to see if it is in acceptable condition",
    "What is this medicine used for?": "You can check the medicine's properties and uses on the back of the label",
    "What are the side effects of this medicine?": "the side effects of the medicine will be mentioned at the bottom right corner of the label",
    "How should I take this medicine?": "If it is in a bottle, consume the medicine using a spoon and if it is a tablet, take it with a glass of water ready",
    "Can I take this medicine on an empty stomach?": "Doctors strongly adivise against taking medicine on an empty stomach as it can be harmful to the body",
    "What should I do if I miss a dose?": "It is acceptable if you miss a dose but please be sure to take your medicine as soon as you remember or if you like, you can also set up an alarm to remind you when to take your medicine",
    "How often should I take this medicine?": "Please consult with your doctor or check your prescription to see how often your should take your medicine",
    "Can I take this medicine with other medications?" : "Please consult with your doctor if you are allowed to take this medicine with other medications at the same time",
    "What is the recommended dosage for this medicine?": "The dosage for the medicine you are required to take should be mentioned on your prescription",
    "Is this medicine safe during pregnancy?": "Please consult with your doctor if it is safe to consume this medicine during pregnancy as the medicine may prove to be harmful to the fetus",
    "How should I store this medicine?": "It is advised to store your medicine in a cool, dark place",
    "What is the shelf life/expiry of this medicine?": "The shelf life/expiry of a particular medicine will be mentioned on the label",
    "Can I drink alcohol while taking this medicine?": "It is generally not recommended to drink alcohol while consuming medicine",
    "Are there any food restrictions with this medicine?": "Yo can consult any food restrictions with your doctor",
    "Can children take this medicine?": "This medicine is not prescribed for children below the age of 5",
    "Is this medicine addictive?": "The medicine is completely safe for use and does not create addictions",
    "Do you offer medicine reminders?": "Yes you can setup an alarm to remind you when it's time to take your medicine",
    "How can I contact customer support?": "The phone number and email for contacting customer support is mentioned in the user dashboard",
    "Do you provide lab tests or health checkup packages?": "Currently we do not provide lab tests or health checkup packages",
    "Can I set up automatic refills for my medicines?": "Yes you can set up an alarm to remind you when to restock your medicine",
    "Is my medical data kept private and secure?": "Yes your medical data is completely private and only accessible to you and your pharmacist",
}

def faq_response(question):
    return faq_dict.get(question, "Sorry, I don’t have an answer for that yet.")

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue='green'), title='PharmaBot - Multilingual OCR') as demo:
    gr.Markdown('# PharmaBot\nMultilingual Prescription Assistant (English + Hindi).')

    with gr.Tab('Chat Interface'):
        chatbot = gr.HTML("<div id='chatbot-box'></div>")
        with gr.Row():
            user_input = gr.Textbox(label='Type your message...', placeholder='Ask about your medicines, dosage, or reminders...', scale=10)
            send_btn = gr.Button('Send', variant='primary', scale=1)
        send_btn.click(update_chat, inputs=[chatbot, user_input], outputs=[chatbot, user_input])

    with gr.Tab('Voice & Image Assistant'):
        with gr.Row():
            audio_in = gr.Audio(sources=['microphone','upload'], type='filepath', label="Patient's Voice")
            image_in = gr.Image(sources=['upload','webcam'], type='filepath', label='Upload / Capture Prescription')
        strip_size_input = gr.Dropdown(choices=[8,10,14,20,28,30], value=10, label='Pills per Strip')
        process_btn = gr.Button('Process Prescription & Voice', variant='primary')
        with gr.Accordion('Results', open=True):
            stt_out = gr.Textbox(label='Transcribed Voice')
            doc_response = gr.Textbox(label="Pharmacist's Response")
            reminder_out = gr.Textbox(label='Medicine Reminder')
            restock_out = gr.Textbox(label='Restock Alert')
            audio_out = gr.Audio(label="Pharmacist's Voice")
        process_btn.click(process_inputs, inputs=[audio_in, image_in, strip_size_input], outputs=[stt_out, doc_response, reminder_out, restock_out, audio_out])

    with gr.Tab('Prescription Reader'):
        image_reader = gr.Image(sources=['upload','webcam'], type='filepath', label='Upload / Capture Prescription')
        reader_out = gr.Textbox(label='Extracted Prescription Text')
        process_reader_btn = gr.Button('Process Prescription', variant='primary')
        process_reader_btn.click(lambda img: extract_prescription_info(img)[0], inputs=image_reader, outputs=reader_out)

        with gr.Column():
            gr.Markdown('### Capture via Colab Webcam (optional)')
            cam_btn = gr.Button('Open Camera & Capture Photo')
            cam_output = gr.Image(label='Captured Image Preview')
            cam_text = gr.Textbox(label='Extracted Text from Captured Prescription')
            def capture_and_read():
                img_path = capture_from_webcam()
                text, _ = extract_prescription_info(img_path)
                return img_path, text
            cam_btn.click(capture_and_read, None, [cam_output, cam_text])

    with gr.Tab('FAQ'):
        faq_dropdown = gr.Dropdown(choices=list(faq_dict.keys()), label='Select a Question')
        faq_answer = gr.Textbox(label='Pharmacist Answer')
        faq_dropdown.change(faq_response, inputs=faq_dropdown, outputs=faq_answer)

demo.launch(share=True, debug=True)
