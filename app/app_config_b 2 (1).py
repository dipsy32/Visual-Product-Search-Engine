
import os, pickle
import hnswlib, numpy as np
import open_clip, streamlit as st
import torch, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from transformers import (
    SegformerImageProcessor,
    AutoModelForSemanticSegmentation,
)

CKPT_DIR = '/kaggle/input/notebooks/gullapellyshrujan/yolo-best/checkpoints'
IMG_ROOT  = '/kaggle/input/notebooks/gullapellyshrujan/yolo-best/checkpoints/cropped_images'

def index_path(alpha, seed):
    return os.path.join(CKPT_DIR, f'hnsw_index_B_alpha{alpha}_seed{seed}.bin')

def meta_path(alpha, seed):
    return os.path.join(CKPT_DIR, f'gallery_meta_B_alpha{alpha}_seed{seed}.pkl')

# ── Model IDs ──────────────────────────────────────────────────────────────────
SEGFORMER_ID = 'mattmdjaga/segformer_b2_clothes'
BLIP2_ID     = 'Salesforce/blip2-opt-2.7b'   # only loaded on demand
CLIP_NAME    = 'ViT-B-32'
CLIP_PRE     = 'openai'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
CLIP_MEAN    = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD     = [0.26862954, 0.26130258, 0.27577711]

# ── SegFormer label map ────────────────────────────────────────────────────────
SEG_LABELS = {
    0:  'Background',  1:  'Hat',           2:  'Hair',
    3:  'Sunglasses',  4:  'Upper-clothes',  5:  'Skirt',
    6:  'Pants',       7:  'Dress',          8:  'Belt',
    9:  'Left-shoe',   10: 'Right-shoe',     11: 'Face',
    12: 'Left-leg',    13: 'Right-leg',      14: 'Left-arm',
    15: 'Right-arm',   16: 'Bag',            17: 'Scarf',
}
CLOTHING_IDS = {1, 4, 5, 6, 7, 8, 9, 10, 16, 17}
LABEL_COLOURS = {
    1:  (245,158, 11),  4:  ( 59,130,246),  5:  (168, 85,247),
    6:  ( 16,185,129),  7:  (236, 72,153),  8:  (239, 68, 68),
    9:  ( 20,184,166), 10:  ( 20,184,166), 16:  (251,191, 36),
    17: ( 99,102,241),
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Visual Product Search — Config B', page_icon='👗', layout='wide')
st.title('👗 Visual Product Search Engine')
st.caption('**Config B** · Frozen CLIP (visual-only, α=0) · SegFormer item-level crop · HNSW retrieval')

# ── Model loaders ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading SegFormer clothing parser …')
def load_segformer():
    processor = SegformerImageProcessor.from_pretrained(
        SEGFORMER_ID, do_reduce_labels=False
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_ID)
    return model.to(DEVICE).eval(), processor

@st.cache_resource(show_spinner='Loading CLIP (frozen) …')
def load_clip():
    model, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRE)
    tokenizer   = open_clip.get_tokenizer(CLIP_NAME)
    model       = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    return model, tokenizer, transform

# BLIP-2 is heavy (~5 GB on CPU). Only loaded when the user explicitly opts in.
@st.cache_resource(show_spinner='Loading BLIP-2 captioner (one-time, ~30 s) …')
def load_blip2():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    dtype     = torch.float16 if DEVICE == 'cuda' else torch.float32
    processor = Blip2Processor.from_pretrained(BLIP2_ID)
    model     = Blip2ForConditionalGeneration.from_pretrained(
                    BLIP2_ID, torch_dtype=dtype).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor

@st.cache_resource(show_spinner='Loading HNSW index …')
def load_index(alpha, seed):
    idx_file  = index_path(alpha, seed)
    meta_file = meta_path(alpha, seed)

    if not os.path.exists(idx_file):
        return None, None, f'Index file not found:\n{idx_file}'
    if not os.path.exists(meta_file):
        return None, None, f'Meta file not found:\n{meta_file}'

    with open(meta_file, 'rb') as f:
        gallery_meta = pickle.load(f)

    # Fix keys (your metadata format)
    gallery_meta['img_paths'] = gallery_meta['gal_paths']
    gallery_meta['item_ids']  = gallery_meta['gal_item_ids']

    dim = gallery_meta['dim']

    index = hnswlib.Index(space='cosine', dim=dim)
    index.load_index(idx_file, max_elements=len(gallery_meta['img_paths']))

    index.set_ef(64)

    return index, gallery_meta, None

# ── SegFormer: detect clothing items → tight crops ─────────────────────────────
def detect_clothing_items(img_pil, seg_model, seg_processor):
    W, H   = img_pil.size
    inputs = seg_processor(images=img_pil, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    upsampled = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
    seg_map   = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

    items = {}
    for label_id in CLOTHING_IDS:
        mask = (seg_map == label_id)
        npx  = int(mask.sum())
        if npx < 500:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        pad  = 12
        x1 = max(0,   int(cols[0])  - pad)
        y1 = max(0,   int(rows[0])  - pad)
        x2 = min(W,   int(cols[-1]) + pad)
        y2 = min(H,   int(rows[-1]) + pad)
        lname = 'Shoes' if label_id in (9, 10) else SEG_LABELS[label_id]
        if lname in items:
            if npx > items[lname][2]:
                items[lname] = (img_pil.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), npx)
        else:
            items[lname] = (img_pil.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), npx)

    items = dict(sorted(items.items(), key=lambda kv: kv[1][2], reverse=True))
    return {k: v[:2] for k, v in items.items()}

def draw_detections(img_pil, detected_items, selected_label):
    preview  = img_pil.copy().convert('RGBA')
    overlay  = Image.new('RGBA', preview.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)
    fallback = [(239,68,68),(59,130,246),(16,185,129),(245,158,11),(168,85,247)]
    fc_iter  = iter(fallback * 5)

    for label, (_, bbox) in detected_items.items():
        lid    = next((k for k, v in SEG_LABELS.items()
                       if v == label or (label == 'Shoes' and k in (9, 10))), None)
        colour = LABEL_COLOURS.get(lid, next(fc_iter))
        is_sel = (label == selected_label)
        draw.rectangle(bbox,
                       fill=colour + (160 if is_sel else 55,),
                       outline=colour + (230,),
                       width=4 if is_sel else 2)

    result   = Image.alpha_composite(preview, overlay).convert('RGB')
    d2       = ImageDraw.Draw(result)
    fc_iter2 = iter(fallback * 5)
    for label, (_, bbox) in detected_items.items():
        lid    = next((k for k, v in SEG_LABELS.items()
                       if v == label or (label == 'Shoes' and k in (9, 10))), None)
        colour = LABEL_COLOURS.get(lid, next(fc_iter2))
        tag    = f'★ {label}' if label == selected_label else label
        d2.text((bbox[0] + 4, max(0, bbox[1] - 14)), tag, fill=colour)
    return result

# ── CLIP + optional BLIP-2 helpers ─────────────────────────────────────────────
def generate_caption(pil_img, blip2_model, blip2_proc):
    dtype  = torch.float16 if DEVICE == 'cuda' else torch.float32
    inputs = blip2_proc(images=pil_img, return_tensors='pt').to(DEVICE)
    inputs = {k: v.to(dtype) if v.is_floating_point() else v for k, v in inputs.items()}
    with torch.no_grad():
        out = blip2_model.generate(**inputs, max_new_tokens=30)
    return blip2_proc.decode(out[0], skip_special_tokens=True).strip()

@torch.no_grad()
def embed_pil(pil_img, clip_model, tokenizer, transform, alpha=1.0, caption=None):
    """Fused CLIP embedding: alpha * visual + (1-alpha) * text(caption)."""
    tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)
    vis_emb = F.normalize(clip_model.encode_image(tensor).float(), dim=-1)
    if alpha >= 1.0 or not caption:
        return vis_emb.cpu().numpy().astype(np.float32)
    tokens  = tokenizer([caption]).to(DEVICE)
    txt_emb = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
    fused   = F.normalize(alpha * vis_emb + (1.0 - alpha) * txt_emb, dim=-1)
    return fused.cpu().numpy().astype(np.float32)

def resolve_gallery_img(stored_path):
    full = os.path.join(IMG_ROOT, stored_path)
    if os.path.exists(full):
        return full
    for i, part in enumerate(stored_path.replace('\\', '/').split('/')):
        if part in ('WOMEN', 'MEN'):
            candidate = os.path.join(IMG_ROOT, *stored_path.replace('\\','/').split('/')[i:])
            if os.path.exists(candidate):
                return candidate
    return full

# ── Load lightweight models only (SegFormer + CLIP) ───────────────────────────
with st.spinner('Loading models — first run only, please wait …'):
    try:
        seg_model,  seg_proc             = load_segformer()
        clip_model, tokenizer, transform = load_clip()
        # BLIP-2 is NOT loaded here — see sidebar toggle below
    except Exception as _load_err:
        st.error(f'Model loading failed: {_load_err}')
        st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️  Settings')

    use_blip2 = st.toggle(
        '🧠 Enable BLIP-2 caption fusion',
        value=False,
        help='Loads the 2.7B BLIP-2 model for text-image fusion. '
             'Disable for much faster startup & inference (pure visual CLIP).'
    )
    if use_blip2:
        st.info('BLIP-2 will be loaded on first search (~30 s one-time cost).')
    else:
        st.success('⚡ Fast mode — pure visual CLIP (no BLIP-2)')

    index_alpha = st.selectbox(
        'Index file α (must match pre-built index)', [0.5, 0.7], index=0,
        help='Selects which pre-built HNSW index file to load.'
    )

    # Fusion slider only meaningful when BLIP-2 is enabled
    if use_blip2:
        fusion_alpha = st.slider(
            'Fusion weight α', 0.0, 1.0, 0.7, 0.05,
            help='1.0 = pure visual CLIP · 0.0 = pure text CLIP · 0.7 recommended'
        )
    else:
        fusion_alpha = 1.0   # pure visual — BLIP-2 never called

    seed  = st.selectbox('HNSW index seed', [599, 600, 124, 605], index=0)
    k_val = st.slider('Number of results  K', 1, 20, 10)
    st.divider()
    st.caption('DeepFashion In-Shop — VR Course Project · Config B')

index, gallery_meta, idx_error = load_index(index_alpha, seed)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.subheader('Step 1 — Upload a Query Image')
uploaded = st.file_uploader(
    'Upload any fashion / person image (JPG / PNG)',
    type=['jpg', 'jpeg', 'png']
)
if uploaded is None:
    st.info(
        '👆 Upload any image — e.g. a rider, street-style shot, product photo. '
        'SegFormer will detect every clothing item (jacket, pants, shoes …) automatically.'
    )
    st.stop()

query_img = Image.open(uploaded).convert('RGB')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SegFormer detection
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.subheader('Step 2 — Clothing Item Detection  (SegFormer)')

with st.spinner('Detecting clothing items …'):
    detected_items = detect_clothing_items(query_img, seg_model, seg_proc)

if not detected_items:
    st.warning('No clothing items detected — using full image as fallback.')
    detected_items = {'Full image': (query_img, (0, 0, *query_img.size))}

col_orig, col_overlay = st.columns(2)
with col_orig:
    st.image(query_img, caption='Uploaded query image', use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Select item
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.subheader('Step 3 — Select Clothing Item to Search For')

selected_label = st.radio(
    'Which item?',
    options=list(detected_items.keys()),
    horizontal=True,
    label_visibility='collapsed'
)

with col_overlay:
    st.image(draw_detections(query_img, detected_items, selected_label),
             caption='Detected items  (★ = selected)', use_container_width=True)

st.markdown('**All detected clothing items:**')
crop_cols = st.columns(min(len(detected_items), 6))
for i, (label, (crop, _)) in enumerate(detected_items.items()):
    with crop_cols[i % 6]:
        is_sel  = (label == selected_label)
        _border = '3px solid #6366f1' if is_sel else '1px solid #2d2d3d'
        _star   = '★ ' if is_sel else ''
        st.markdown(
            f'<div style="border:{_border};border-radius:8px;padding:3px;">'
            f'<p style="text-align:center;font-size:11px;color:#94a3b8;margin:2px 0;">'
            f'{_star}{label}</p></div>',
            unsafe_allow_html=True
        )
        st.image(crop, use_container_width=True)

selected_crop, selected_bbox = detected_items[selected_label]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Confirm crop
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.subheader('Step 4 — Confirm Crop')

col_c1, col_c2 = st.columns([1, 2])
with col_c1:
    st.image(selected_crop, caption=f'Auto crop: {selected_label}', use_container_width=True)
with col_c2:
    action = st.radio(
        'Does the crop look correct?',
        ['✅ Use this crop', '🔁 Use full image', '✏️ Re-crop manually'],
        index=0
    )
    if 'full' in action:
        search_img = query_img
    elif 'manually' in action:
        W, H = query_img.size
        x1d, y1d, x2d, y2d = selected_bbox
        cx1 = st.slider('Left  (x1)', 0, W-1, x1d)
        cx2 = st.slider('Right (x2)', 1, W,   x2d)
        cy1 = st.slider('Top   (y1)', 0, H-1, y1d)
        cy2 = st.slider('Bottom(y2)', 1, H,   y2d)
        search_img = query_img.crop((cx1, cy1, cx2, cy2))
        st.image(search_img, caption='Manual crop preview', use_container_width=True)
    else:
        search_img = selected_crop

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Search
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.subheader('Step 5 — Search')

if st.button('🔍 Search Similar Products', type='primary'):

    if index is None:
        st.error(idx_error, icon='❌')
        st.stop()

    caption_text = None

    # Only invoke BLIP-2 when the user opted in AND fusion would actually use it
    if use_blip2 and fusion_alpha < 1.0:
        with st.spinner('Loading BLIP-2 & generating caption …'):
            blip2_model, blip2_proc = load_blip2()
            caption_text = generate_caption(search_img, blip2_model, blip2_proc)
        st.info(f'🖊️ **BLIP-2 caption:** {caption_text}')
    else:
        st.info('⚡ Skipping BLIP-2 — pure visual CLIP embedding.')

    with st.spinner('Encoding with CLIP …'):
        q_emb = embed_pil(search_img, clip_model, tokenizer, transform,
                          fusion_alpha, caption_text)

    labels_result, distances = index.knn_query(q_emb, k=k_val)
    labels_result = labels_result[0]
    distances     = distances[0]

    gal_item_ids    = gallery_meta['item_ids']
    gal_paths       = gallery_meta['img_paths']
    retrieved_ids   = [gal_item_ids[i] for i in labels_result]
    retrieved_paths = [gal_paths[i]    for i in labels_result]
    retrieved_sims  = (1.0 - distances).tolist()

    st.markdown('---')
    st.subheader(f'Top-{k_val} Results')

    for row_start in range(0, k_val, 5):
        cols = st.columns(5)
        for col_idx, rank in enumerate(range(row_start, min(row_start + 5, k_val))):
            img_path = resolve_gallery_img(retrieved_paths[rank])
            item_id  = retrieved_ids[rank]
            sim      = retrieved_sims[rank]
            _colors  = ['black','white','red','blue','green','yellow','pink',
                        'purple','orange','grey','brown','beige','navy','cream']
            color    = next((c for c in _colors if c in item_id.lower()), '—')
            with cols[col_idx]:
                if os.path.exists(img_path):
                    st.image(Image.open(img_path).convert('RGB'), use_container_width=True)
                else:
                    st.warning(f'Not found:\n{img_path}')
                st.markdown(
                    f'**Rank {rank+1}**\n'
                    f'Item: `{item_id}`\n'
                    f'Sim: `{sim:.4f}`\n'
                    f'Color: {color}'
                )

    st.success(f'Search complete — top {k_val} results shown.')
