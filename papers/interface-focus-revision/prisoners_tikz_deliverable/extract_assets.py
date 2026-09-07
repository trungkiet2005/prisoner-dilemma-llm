from pathlib import Path
import cv2
import numpy as np
from PIL import Image

SRC = Path('/mnt/data/e85d993c-ac76-4d58-95d1-888da93525c4.png')
OUT = Path('/mnt/data/tikz_prisoners/assets')
OUT.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(str(SRC), cv2.IMREAD_COLOR)
if img_bgr is None:
    raise FileNotFoundError(SRC)

# Coordinates are from the supplied 1448x1086 raster.
SIMPLE = {
    'logo_claude.png': (55, 696, 130, 758),
    'logo_openai.png': (191, 696, 254, 758),
    'logo_gemini.png': (323, 693, 399, 762),
    'logo_qwen.png': (457, 696, 532, 760),
    'logo_xai.png': (589, 696, 657, 759),
    'icon_same_provider.png': (47, 895, 170, 958),
    'icon_no_communication.png': (223, 891, 330, 958),
    'icon_languages.png': (402, 893, 485, 959),
    'icon_personas.png': (530, 892, 678, 961),
    'icon_payoff_scales.png': (765, 892, 850, 959),
    'icon_corpus.png': (986, 890, 1072, 960),
    'icon_theory.png': (1200, 897, 1262, 966),
}


def save_rgba(name, rgba, trim=True, pad=3, upscale=2):
    alpha = rgba[:, :, 3]
    if trim:
        ys, xs = np.where(alpha > 8)
        if len(xs):
            x1, x2 = max(0, xs.min()-pad), min(rgba.shape[1], xs.max()+pad+1)
            y1, y2 = max(0, ys.min()-pad), min(rgba.shape[0], ys.max()+pad+1)
            rgba = rgba[y1:y2, x1:x2]
    pil = Image.fromarray(rgba, mode='RGBA')
    if upscale and upscale > 1:
        pil = pil.resize((pil.width*upscale, pil.height*upscale), Image.Resampling.LANCZOS)
    pil.save(OUT/name)


def remove_flat_background(box):
    x1,y1,x2,y2 = box
    crop = img_bgr[y1:y2, x1:x2].copy()
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    h,w = rgb.shape[:2]
    # Estimate local card background from an outer border.
    bw = max(3, min(h,w)//12)
    border = np.concatenate([
        rgb[:bw,:,:].reshape(-1,3), rgb[-bw:,:,:].reshape(-1,3),
        rgb[:, :bw,:].reshape(-1,3), rgb[:, -bw:,:].reshape(-1,3)
    ], axis=0)
    # Robustly use the brightest 60% of border pixels as background samples.
    brightness = border.mean(axis=1)
    bg_samples = border[brightness >= np.percentile(brightness, 40)]
    bg = np.median(bg_samples, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32)-bg[None,None,:], axis=2)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32)
    val = hsv[:,:,2].astype(np.float32)
    # Background separation: distance, saturation and darkness all contribute.
    alpha_dist = np.clip((dist-8.0)/22.0, 0, 1)
    alpha_sat = np.clip((sat-18.0)/55.0, 0, 1)
    alpha_dark = np.clip((238.0-val)/65.0, 0, 1)
    alpha = np.maximum.reduce([alpha_dist, alpha_sat, alpha_dark])
    # Strongly suppress nearly-white/light-blue card background.
    light_neutral = (val > 226) & (sat < 42) & (dist < 34)
    alpha[light_neutral] *= np.clip((dist[light_neutral]-6)/22, 0, 1)
    alpha = cv2.GaussianBlur((alpha*255).astype(np.uint8), (3,3), 0.55)
    rgba = np.dstack([rgb, alpha])
    return rgba


for name, box in SIMPLE.items():
    rgba = remove_flat_background(box)
    save_rgba(name, rgba, trim=True, pad=2, upscale=2)


def grab_caged():
    # Use the large left cage for better resolution.
    x1,y1,x2,y2 = (40, 268, 282, 614)
    crop = img_bgr[y1:y2, x1:x2].copy()
    h,w = crop.shape[:2]
    mask = np.full((h,w), cv2.GC_PR_BGD, np.uint8)
    b=5
    mask[:b,:] = cv2.GC_BGD; mask[-b:,:] = cv2.GC_BGD
    mask[:,:b] = cv2.GC_BGD; mask[:,-b:] = cv2.GC_BGD
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:,:,1], hsv[:,:,2]
    strong = ((sat>70)&(val>65)) | (val<165)
    mask[strong] = cv2.GC_FGD
    # Force the cage and robot interior as foreground; surrounding room remains probable BG.
    mask[16:h-16, 18:w-18] = np.where(mask[16:h-16,18:w-18] == cv2.GC_FGD, cv2.GC_FGD, cv2.GC_PR_FGD)
    # Definite robot body/head to keep the white shell.
    mask[105:245, 55:187] = cv2.GC_FGD
    mask[210:302, 62:180] = cv2.GC_FGD
    bgd = np.zeros((1,65), np.float64); fgd = np.zeros((1,65), np.float64)
    cv2.grabCut(crop, mask, None, bgd, fgd, 7, cv2.GC_INIT_WITH_MASK)
    m = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),1,0).astype(np.uint8)
    # Keep large connected components; cage has multiple pieces connected by top/bottom rails.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1,n):
        if stats[i, cv2.CC_STAT_AREA] > 60:
            keep[labels==i] = 1
    alpha = cv2.GaussianBlur((keep*255).astype(np.uint8),(3,3),0.45)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return np.dstack([rgb,alpha])


def grab_free():
    # Top-right free robot in payoff matrix.
    x1,y1,x2,y2 = (1300, 325, 1415, 480)
    crop = img_bgr[y1:y2, x1:x2].copy()
    h,w = crop.shape[:2]
    mask = np.full((h,w), cv2.GC_PR_BGD, np.uint8)
    b=4
    mask[:b,:] = cv2.GC_BGD; mask[-b:,:] = cv2.GC_BGD
    mask[:,:b] = cv2.GC_BGD; mask[:,-b:] = cv2.GC_BGD
    samples=np.concatenate([crop[:15,:15].reshape(-1,3), crop[:15,-15:].reshape(-1,3),
                            crop[-15:,:15].reshape(-1,3), crop[-15:,-15:].reshape(-1,3)],axis=0)
    bg=np.median(samples,axis=0)
    dist=np.linalg.norm(crop.astype(np.float32)-bg[None,None,:],axis=2)
    yy,xx=np.ogrid[:h,:w]
    core=((xx>w*0.08)&(xx<w*0.96)&(yy>h*0.24)&(yy<h*0.95))
    mask[(dist<16)&(~core)] = cv2.GC_BGD
    mask[(dist<28)&(~core)] = cv2.GC_PR_BGD
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV); sat=hsv[:,:,1]; val=hsv[:,:,2]
    strong=((sat>90)&(val>80))|(val<170)
    mask[strong]=cv2.GC_FGD
    mask[int(h*.37):int(h*.72), int(w*.10):int(w*.90)] = cv2.GC_FGD
    mask[int(h*.67):int(h*.93), int(w*.18):int(w*.82)] = cv2.GC_FGD
    bgd=np.zeros((1,65),np.float64); fgd=np.zeros((1,65),np.float64)
    cv2.grabCut(crop,mask,None,bgd,fgd,8,cv2.GC_INIT_WITH_MASK)
    m=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),1,0).astype(np.uint8)
    n,lab,stats,cents=cv2.connectedComponentsWithStats(m,8)
    keep=np.zeros_like(m)
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA]; cx,cy=cents[i]
        if area>20 and abs(cx-w/2)<w*.55 and abs(cy-h*.55)<h*.55:
            keep[lab==i]=1
    alpha=cv2.GaussianBlur((keep*255).astype(np.uint8),(3,3),0.45)
    rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    return np.dstack([rgb,alpha])

save_rgba('robot_caged.png', grab_caged(), trim=True, pad=2, upscale=2)
save_rgba('robot_free.png', grab_free(), trim=True, pad=2, upscale=3)

print(f'Wrote {len(SIMPLE)+2} transparent PNG assets to {OUT}')
