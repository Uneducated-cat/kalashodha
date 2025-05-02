import os
import re
import uuid
import json
from pathlib import Path
from PIL import Image, ImageEnhance
import gradio as gr
import ollama
import numpy as np
from sklearn.cluster import KMeans

# ---- User Management ----
USER_DB = Path("users.json")
if not USER_DB.exists():
    USER_DB.write_text(json.dumps({}))

def get_users():
    try:
        return json.loads(USER_DB.read_text())
    except Exception:
        return {}

def save_users(users):
    USER_DB.write_text(json.dumps(users))

def signup(username, password):
    users = get_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {"password": password, "tier": "free"}
    save_users(users)
    return True, "Signup successful! Please log in."

def login(username, password):
    users = get_users()
    if username not in users:
        return False, "User does not exist."
    if users[username]["password"] != password:
        return False, "Incorrect password."
    return True, "Login successful!"

def upgrade_account(username):
    users = get_users()
    if username in users:
        users[username]["tier"] = "premium"
        save_users(users)
        return True, "Account upgraded to premium!"
    return False, "User not found."

def get_user_tier(username):
    users = get_users()
    return users.get(username, {}).get("tier", "free")

# ---- Palette Extraction (KMeans) ----
def extract_palette(image, num_colors=6):
    img = image.convert('RGB')
    img_np = np.array(img)
    pixels = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init=5, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    palette = [tuple(color) for color in colors]
    return palette

def create_swatch(rgb):
    swatch = Image.new('RGB', (60, 60), rgb)
    return swatch

# ---- Style Presets ----
STYLE_PRESETS = {
    "Vintage": {"contrast": 0.9, "saturation": 0.7, "hue_shift": 20},
    "Monochrome": {"saturation": 0, "contrast": 1.2},
    "Watercolor": {"sharpness": 0.5, "brightness": 1.1}
}

# ---- Feedback, Parsing, Adjustment ----
def get_feedback(image_path):
    try:
        response = ollama.chat(
            model="llava:13b",
            messages=[{
                'role': 'user',
                'content': "Analyze this artwork. Provide specific technical adjustments with percentages "
                           "(e.g., 'Contrast +20%', 'Reduce blue by 15%'). Include multiple suggestions per category. "
                           "Also identify art style and historical context. Use clear formatting.",
                'images': [image_path]
            }]
        )
        feedback = response['message']['content']
        return feedback
    except Exception as e:
        return f"Error: {str(e)}"

def parse_feedback(feedback):
    adjustments = {
        'contrast': 1.0,
        'brightness': 1.0,
        'saturation': 1.0,
        'sharpness': 1.0,
        'hue_shift': 0.0,
        'color': {'red': 1.0, 'green': 1.0, 'blue': 1.0}
    }
    patterns = [
        (r'(contrast|brightness|saturation|sharpness)\s*([+-])\s*(\d+\.?\d*)%', 'property_sign'),
        (r'(red|green|blue)\s*([+-])\s*(\d+\.?\d*)%', 'color_sign'),
        (r'(hue|hue shift)\s*([+-])\s*(\d+\.?\d*)%', 'hue_sign'),
        (r'(increase|boost|raise|enhance)\s+(contrast|brightness|saturation|sharpness|red|green|blue|hue|hue shift)[^\d]*(\d+\.?\d*)%', 'verb_increase'),
        (r'(decrease|reduce|lower|dim|soften)\s+(contrast|brightness|saturation|sharpness|red|green|blue|hue|hue shift)[^\d]*(\d+\.?\d*)%', 'verb_decrease'),
    ]
    for pattern, mode in patterns:
        matches = re.findall(pattern, feedback, re.IGNORECASE)
        for match in matches:
            if mode == 'property_sign':
                prop, sign, value = match
                factor = 1 + (float(value)/100.0 if sign == '+' else -float(value)/100.0)
                prop = prop.lower()
                if prop in adjustments:
                    adjustments[prop] *= factor
                    adjustments[prop] = max(0.5, min(adjustments[prop], 2.0))
            elif mode == 'color_sign':
                channel, sign, value = match
                factor = 1 + (float(value)/100.0 if sign == '+' else -float(value)/100.0)
                channel = channel.lower()
                if channel in adjustments['color']:
                    adjustments['color'][channel] *= factor
                    adjustments['color'][channel] = max(0.5, min(adjustments['color'][channel], 2.0))
            elif mode == 'hue_sign':
                _, sign, value = match
                shift = float(value) if sign == '+' else -float(value)
                adjustments['hue_shift'] += shift
            elif mode == 'verb_increase':
                _, prop, value = match
                prop = prop.lower()
                if prop in adjustments:
                    adjustments[prop] *= 1 + float(value)/100.0
                    adjustments[prop] = max(0.5, min(adjustments[prop], 2.0))
                elif prop in adjustments['color']:
                    adjustments['color'][prop] *= 1 + float(value)/100.0
                    adjustments['color'][prop] = max(0.5, min(adjustments['color'][prop], 2.0))
                elif 'hue' in prop:
                    adjustments['hue_shift'] += float(value)
            elif mode == 'verb_decrease':
                _, prop, value = match
                prop = prop.lower()
                if prop in adjustments:
                    adjustments[prop] *= 1 - float(value)/100.0
                    adjustments[prop] = max(0.5, min(adjustments[prop], 2.0))
                elif prop in adjustments['color']:
                    adjustments['color'][prop] *= 1 - float(value)/100.0
                    adjustments['color'][prop] = max(0.5, min(adjustments['color'][prop], 2.0))
                elif 'hue' in prop:
                    adjustments['hue_shift'] -= float(value)
    return adjustments

def apply_adjustments(image_path, adjustments):
    img = Image.open(image_path).convert('RGB')
    img = ImageEnhance.Contrast(img).enhance(adjustments['contrast'])
    img = ImageEnhance.Brightness(img).enhance(adjustments['brightness'])
    img = ImageEnhance.Color(img).enhance(adjustments['saturation'])
    img = ImageEnhance.Sharpness(img).enhance(adjustments['sharpness'])
    r, g, b = img.split()
    r = r.point(lambda i: i * adjustments['color']['red'])
    g = g.point(lambda i: i * adjustments['color']['green'])
    b = b.point(lambda i: i * adjustments['color']['blue'])
    img = Image.merge('RGB', (r, g, b))
    if abs(adjustments.get('hue_shift', 0.0)) > 0.1:
        img = hue_shift(img, adjustments['hue_shift'])
    return img

def hue_shift(img, degrees):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.array(img.convert('HSV')).astype(np.uint16)
    hue_shift = int((degrees / 360.0) * 255)
    arr[..., 0] = (arr[..., 0] + hue_shift) % 256
    arr = arr.astype('uint8')
    return Image.fromarray(arr, 'HSV').convert('RGB')

# ---- Main Process ----
def process_image(image, username):
    if image is None:
        return "No image provided", None, [None]*6, []
    temp_input_path = f"temp_input_{uuid.uuid4().hex[:8]}.jpg"
    image.save(temp_input_path, quality=95)
    feedback = get_feedback(temp_input_path)
    if feedback.startswith("Error:"):
        os.remove(temp_input_path)
        return feedback, None, [None]*6, []
    adjustments = parse_feedback(feedback)
    adjusted_img = apply_adjustments(temp_input_path, adjustments)
    user_tier = get_user_tier(username) if username else "free"
    if user_tier == "premium":
        palette = extract_palette(image)
        swatches = [create_swatch(rgb) for rgb in palette]
        styles_to_show = list(STYLE_PRESETS.keys())
        style_images = []
        for style in styles_to_show:
            style_adj = adjustments.copy()
            for k, v in STYLE_PRESETS[style].items():
                if k == "color":
                    style_adj["color"].update(v)
                else:
                    style_adj[k] = v
            style_img = apply_adjustments(temp_input_path, style_adj)
            style_images.append((style_img, style))
        os.remove(temp_input_path)
        return feedback, adjusted_img, swatches, style_images
    else:
        os.remove(temp_input_path)
        return feedback, adjusted_img, [None]*6, []

# ---- Gradio UI ----
with gr.Blocks(title="Artwork Enhancer AI") as demo:
    gr.Markdown("""
    # 🎨 AI Artwork Enhancer
    Upload an artwork for technical analysis and enhancement.  
    <span style="color:green">Premium users get color palette extraction and style variations.</span>
    """)
    state = gr.State({"username": None})

    # --- Auth Section ---
    with gr.Row():
        with gr.Column():
            login_username = gr.Textbox(label="Username")
            login_password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            signup_btn = gr.Button("Sign Up")
            auth_msg = gr.Markdown("")
            logout_btn = gr.Button("Logout", visible=False)
            upgrade_btn = gr.Button("Upgrade to Premium", visible=False)
            premium_msg = gr.Markdown("", visible=False)
    
    # --- Main App Section ---
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Original Artwork", type="pil", height=400)
            submit_btn = gr.Button("Analyze & Enhance")
        with gr.Column():
            output_image = gr.Image(label="Enhanced Artwork", type="pil", height=400)
            feedback_output = gr.Textbox(label="Technical Analysis", lines=10)
    
    # Palette (hidden by default)
    with gr.Row(visible=False) as palette_row:
        gr.Markdown("**Extracted Color Palette:**")
        swatch_imgs = [gr.Image(label=f"Color {i+1}", width=60, height=60) for i in range(6)]
    
    # Style Gallery (hidden by default)
    with gr.Row(visible=False) as style_row:
        gr.Markdown("**Style Variations:**")
        style_gallery = gr.Gallery(label="Styles", columns=3, object_fit="contain", height=200)

    # --- Logic ---
    def handle_login(username, password, state):
        ok, msg = login(username, password)
        if ok:
            state["username"] = username
            tier = get_user_tier(username)
            show_premium = tier == "premium"
            # Clear all outputs/inputs after login
            return (
                gr.update(visible=True),  # logout_btn
                gr.update(visible=True),  # upgrade_btn
                gr.update(visible=show_premium),  # premium_msg
                f"Welcome, {username}!",
                state,
                gr.update(visible=show_premium),  # palette_row
                gr.update(visible=show_premium),  # style_row
                None,  # feedback_output
                None,  # output_image
                *[None]*6,  # swatch_imgs
                [],  # style_gallery
                None,  # input_image
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                msg,
                state,
                gr.update(visible=False),
                gr.update(visible=False),
                None,
                None,
                *[None]*6,
                [],
                None,
            )

    def handle_signup(username, password, state):
        ok, msg = signup(username, password)
        return msg

    def handle_logout(state):
        state["username"] = None
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "Logged out.",
            state,
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None,
            *[None]*6,
            [],
            None,
        )

    def handle_upgrade(state):
        ok, msg = upgrade_account(state["username"])
        tier = get_user_tier(state["username"])
        show_premium = tier == "premium"
        # Clear all outputs/inputs after upgrade
        return (
            msg,  # premium_msg
            None,  # feedback_output
            None,  # output_image
            *[None]*6,  # swatch_imgs
            [],  # style_gallery
            gr.update(visible=show_premium),  # palette_row
            gr.update(visible=show_premium),  # style_row
            None,  # input_image
        )

    def handle_submit(image, state):
        username = state["username"]
        feedback, enhanced, swatches, style_imgs = process_image(image, username)
        user_tier = get_user_tier(username) if username else "free"
        is_premium = user_tier == "premium"
        swatch_values = swatches if is_premium else [None]*6
        style_gallery_value = style_imgs if is_premium else []
        return (
            feedback,
            enhanced,
            *swatch_values,
            style_gallery_value,
            gr.update(visible=is_premium),  # palette_row
            gr.update(visible=is_premium),  # style_row
        )

    login_btn.click(
        handle_login,
        [login_username, login_password, state],
        [logout_btn, upgrade_btn, premium_msg, auth_msg, state, palette_row, style_row,
         feedback_output, output_image, *swatch_imgs, style_gallery, input_image]
    )
    signup_btn.click(handle_signup, [login_username, login_password, state], auth_msg)
    logout_btn.click(
        handle_logout,
        state,
        [logout_btn, upgrade_btn, premium_msg, auth_msg, state, palette_row, style_row,
         feedback_output, output_image, *swatch_imgs, style_gallery, input_image]
    )
    upgrade_btn.click(
        handle_upgrade,
        state,
        [premium_msg, feedback_output, output_image, *swatch_imgs, style_gallery, palette_row, style_row, input_image]
    )
    submit_btn.click(
        handle_submit,
        [input_image, state],
        [feedback_output, output_image] + swatch_imgs + [style_gallery, palette_row, style_row]
    )

if __name__ == "__main__":
    demo.launch()
