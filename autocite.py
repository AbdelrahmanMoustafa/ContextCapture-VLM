import os
import time
import re
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pygetwindow as gw
import uiautomation as auto
from PIL import Image, PngImagePlugin
import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration

SCREENSHOT_DIR = str(Path.home() / "Pictures" / "Screenshots")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "florence-community/Florence-2-base-ft"

print(f"[*] Loading native VLM model '{MODEL_ID}' on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Florence2ForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
print("[*] VLM model loaded and ready.")

def clean_filename(text: str) -> str:
    if not text or text == "Unknown_App": return "screenshot"
    
    cleaned = text.lower()
    
    fluff_phrases = [
        "a computer screen showing a ", "a computer screen showing ",
        "a screen shot of a ", "a screen shot of ",
        "a screenshot of a ", "a screenshot of ",
        "the image shows a ", "the image shows ",
        "this is a picture of a ", "this is a picture of ",
        "a picture of a ", "a picture of ",
        "a close up of a ", "a close up of ",
        "an image of a ", "an image of ",
        "qa>", "qa"
    ]
    for fluff in fluff_phrases:
        cleaned = cleaned.replace(fluff, "")
        
    for suffix in [" - youtube", " - google chrome", " - microsoft edge", " - mozilla firefox", " - brave"]:
        cleaned = cleaned.replace(suffix, "")
        
    cleaned = cleaned.replace(' ', '_').replace('-', '_')
    cleaned = re.sub(r'[\\/*?:"<>|.\'\[\]\(\)]', "", cleaned)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    
    return cleaned[:60]

def get_smart_description(image_path: str, window_title: str):
    try:
        image = Image.open(image_path).convert("RGB")
        
        task_prompt = "<CAPTION>"
        
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=30, 
                num_beams=3,
                do_sample=False
            )
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ai_output = generated_text.replace(task_prompt, "").strip()
        print(f"    [AI Raw Output] {ai_output}")
        
        cleaned_ai = clean_filename(ai_output)
        
        media_trigger_words = {
            "song", "track", "album", "music", "spotify", "youtube", 
            "video", "movie", "poem", "website", "webpage", "player", 
            "browser", "text", "no", "none", "unknown", "blank", "empty"
        }
        ai_word_list = set(cleaned_ai.split('_'))
        
        # If the AI caption contains a media word, THROW IT AWAY and use the Window Title.
        # Otherwise, it's a general picture (like a gaming setup), so keep the AI description!
        if not cleaned_ai or ai_word_list.intersection(media_trigger_words):
            print(f"    [Smart Fallback] AI detected Media/UI context. Routing to OS Window Title.")
            return clean_filename(window_title)
            
        return cleaned_ai

    except Exception as e:
        print(f"    [-] VLM extraction failed: {e}")
        return clean_filename(window_title)

def get_browser_url(active_window):
    """Attempts to extract the URL if the active window is a browser."""
    try:
        if not active_window: return None
        title = active_window.title
        url = None
        browser_markers = ["Google Chrome", "Microsoft Edge", "Brave", "Mozilla Firefox", "Opera", "Vivaldi"]
        if any(marker.lower() in title.lower() for marker in browser_markers):
            window_handle = getattr(active_window, "_hWnd", None)
            window_control = auto.ControlFromHandle(window_handle) if window_handle else auto.WindowControl(Name=title)
            for address_bar_name in ["Address and search bar", "Search or enter address", "Search the web or type a URL", "Address bar"]:
                address_bar = window_control.EditControl(Name=address_bar_name)
                if address_bar.Exists(0, 0):
                    value = address_bar.GetValuePattern().Value
                    if value:
                        url = value if value.startswith("http") else "https://" + value
                        break
        return url
    except Exception as e:
        return None

class ScreenshotHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return
            
        filepath = event.src_path
        print(f"\n[+] New screenshot detected: {filepath}")
        
        active_window = gw.getActiveWindow()
        window_title = active_window.title if active_window else "Unknown_App"
        
        file_ready = False
        last_size = -1
        for attempt in range(20): 
            try:
                current_size = os.path.getsize(filepath)
                if current_size > 0 and current_size == last_size:
                    file_ready = True
                    break
                last_size = current_size
            except OSError: pass 
            time.sleep(0.5)
            
        if not file_ready:
            print("    [-] Error: File remained locked. Skipping.")
            return

        print(f"    [Processing] Active Window: {window_title}")
        description = get_smart_description(filepath, window_title)
        
        if not description:
            description = clean_filename(window_title)

        url = get_browser_url(active_window) if active_window else None
        if url:
            print(f"    [URL Detected] {url}")
        
        try:
            source_extension = Path(filepath).suffix.lower() or ".png"
            new_filename = f"{description}_{time.time_ns()}{source_extension}"
            new_filepath = os.path.join(os.path.dirname(filepath), new_filename)
            
            if filepath.lower().endswith('.png') and url:
                img = Image.open(filepath)
                meta = PngImagePlugin.PngInfo()
                meta.add_text("Source_URL", url)
                meta.add_text("Source_App", window_title)
                img.save(new_filepath, "png", pnginfo=meta)
                img.close()
                os.remove(filepath)
                print(f"    [v] Success! Saved with EXIF URL -> {new_filename}")
            else:
                os.rename(filepath, new_filepath)
                print(f"    [v] Success! File saved as -> {new_filename}")
                
        except Exception as e:
            print(f"    [-] Failed to process file: {e}")

if __name__ == "__main__":
    if not os.path.exists(SCREENSHOT_DIR):
        exit(1)

    event_handler = ScreenshotHandler()
    observer = Observer()
    observer.schedule(event_handler, SCREENSHOT_DIR, recursive=False)
    observer.start()
    
    print(f"[*] AutoCite Universal VLM is running.")
    print(f"[*] Monitoring: {SCREENSHOT_DIR}")
    print("[*] Press Ctrl+C to stop.\n")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()