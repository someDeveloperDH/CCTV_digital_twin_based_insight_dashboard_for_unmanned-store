"""
CCTV 방문자 분석 파이프라인 v2
변경사항:
  - yolov8n-pose.pt: 키포인트 기반 성별/나이 (랜덤 분류기 제거)
  - FRAME_SKIP=2: 처리 속도 3× 향상
  - FP16: GPU 추론 가속
  - process_camera() → (TrackData, metrics) 반환
실행: python analyze.py [--day 10] [--no-display]
"""
import argparse, cv2, json, math, time, torch, numpy as np, timm
import torchvision.transforms as T
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

BASE_DIR    = Path(__file__).parent
MOSAIC_DIR  = BASE_DIR / "mosaic"
MERGED_DIR  = BASE_DIR / "merged"
ZONES_FILE  = BASE_DIR / "zones.json"
RESULT_FILE = BASE_DIR / "visit_data.json"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL  = "yolov8n-pose.pt"   # pose → 키포인트 제공
USE_FP16    = DEVICE == "cuda"
CONF        = 0.4
IOU         = 0.5
FRAME_SKIP  = 2          # 1프레임 처리 후 2프레임 건너뜀 → 3× 속도
EMBED_EVERY = 3          # 처리된 프레임 기준 임베딩 추출 주기
ATTR_EVERY  = 5          # 처리된 프레임 기준 성별/나이 추출 주기
DISPLAY_SCALE = 0.5

DWELL_MIN_SEC      = 3.0
PURCHASE_DWELL_SEC = 5.0

# Re-ID 가중치 (합=1.0)
W_TIME=0.30; W_COLOR=0.25; W_BODY=0.15; W_APPEAR=0.20; W_SEQ=0.10
TIME_WINDOW_SEC = 300
REID_THRESHOLD  = 0.45

# 카메라 스펙
CAM_HEIGHT_M=2.2; CAM_TILT_DEG=35.0; CAM_VFOV_DEG=40.0
ADULT_MIN_M=1.5;  ADULT_MARGIN=0.85

# COCO 키포인트 인덱스
KP_L_SH, KP_R_SH   = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_ANK, KP_R_ANK = 15, 16
KP_CONF_MIN = 0.45

ZONE_COLORS = [(0,255,0),(255,128,0),(0,128,255),(255,0,128),(128,255,0),
               (0,255,200),(200,0,255),(255,255,0),(0,128,128)]
C_MALE=(255,180,0); C_FEMALE=(0,100,255); C_UNK=(180,180,180); C_BUY=(0,0,255)


# ── 타임라인 ──────────────────────────────────────────────────────────────────

def load_timeline(cam, day):
    path = MERGED_DIR / f"{cam}_{day}_timeline.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        tl = json.load(f)
    for clip in tl["clips"]:
        clip["_s"] = datetime.strptime(clip["time_start"], "%Y-%m-%d %H:%M:%S")
        clip["_e"] = datetime.strptime(clip["time_end"],   "%Y-%m-%d %H:%M:%S")
    return tl


def frame_to_dt(fi, tl):
    if tl is None: return None
    for c in tl["clips"]:
        if c["frame_start"] <= fi <= c["frame_end"]:
            return c["_s"] + timedelta(seconds=(fi - c["frame_start"]) / tl["fps"])
    return None


def get_timeband(entry_time):
    if entry_time is None: return "-"
    h = int(entry_time[:2])
    if h < 12: return "오전"
    if h < 17: return "오후"
    return "저녁"


# ── Re-ID 임베딩 (MobileNetV3 backbone) ──────────────────────────────────────

class ReIDExtractor:
    def __init__(self):
        self.model = timm.create_model(
            "mobilenetv3_small_100", pretrained=True, num_classes=0
        ).to(DEVICE).eval()
        if USE_FP16: self.model = self.model.half()
        self.tf = T.Compose([
            T.ToPILImage(), T.Resize((256,128)), T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    @torch.no_grad()
    def extract(self, crop):
        if crop.shape[0] < 60 or crop.shape[1] < 10: return None
        body = crop[int(crop.shape[0]*0.25):, :]
        rgb  = cv2.cvtColor(body if body.shape[0]>10 else crop, cv2.COLOR_BGR2RGB)
        t    = self.tf(rgb).unsqueeze(0).to(DEVICE)
        if USE_FP16: t = t.half()
        feat = self.model(t).cpu().float().numpy()[0]
        n    = np.linalg.norm(feat)
        return feat/n if n > 0 else feat


# ── 키포인트 기반 성별/나이 ────────────────────────────────────────────────────

def gender_from_keypoints(kpts):
    """
    어깨/골반 너비 비율 → 성별 추정 (바이오메카닉 기반)
    남성: 어깨 넓음 (ratio ≥ 1.08)  여성: 골반 넓음 (ratio ≤ 0.98)
    불확실 구간 (0.98 ~ 1.08) → None
    """
    if kpts is None or len(kpts) < 13: return None, 0.0
    ls,rs = kpts[KP_L_SH],  kpts[KP_R_SH]
    lh,rh = kpts[KP_L_HIP], kpts[KP_R_HIP]
    if any(float(k[2]) < KP_CONF_MIN for k in [ls,rs,lh,rh]): return None, 0.0
    sw = abs(float(rs[0]) - float(ls[0]))
    hw = abs(float(rh[0]) - float(lh[0]))
    if hw < 5: return None, 0.0
    ratio = sw / hw
    if ratio >= 1.08:
        return "M", min(0.82, 0.58 + (ratio-1.08)*1.6)
    if ratio <= 0.98:
        return "F", min(0.82, 0.58 + (0.98-ratio)*1.6)
    return None, 0.38


def adult_from_keypoints(kpts, fh):
    """키포인트 머리~발목 픽셀 거리 → 카메라 기하학으로 실제 키 추정 → 성인 판별"""
    if kpts is None or len(kpts) < 17: return None
    hpts = [kpts[i] for i in [0,1,2,3,4] if float(kpts[i][2]) >= KP_CONF_MIN]
    apts = [kpts[i] for i in [KP_L_ANK,KP_R_ANK] if float(kpts[i][2]) >= KP_CONF_MIN]
    if not hpts or not apts: return None
    y_top = int(min(float(p[1]) for p in hpts))
    y_bot = int(max(float(p[1]) for p in apts))
    if y_bot <= y_top: return None
    return estimate_adult(y_top, y_bot, fh)


# ── 카메라 기하학 ─────────────────────────────────────────────────────────────

def estimate_adult(y1, y2, fh):
    tr = math.radians(CAM_TILT_DEG); vr = math.radians(CAM_VFOV_DEG)
    phi = tr + (y2/fh - 0.5)*vr
    if phi <= 0: return True
    d = max(CAM_HEIGHT_M/math.tan(phi), 0.5)
    L = math.sqrt(CAM_HEIGHT_M**2 + d**2)
    return (y2-y1)/fh >= math.atan(ADULT_MIN_M/L)/vr * ADULT_MARGIN


def body_proportion(y1, y2, fh):
    tr = math.radians(CAM_TILT_DEG); vr = math.radians(CAM_VFOV_DEG)
    phi = tr + (y2/fh - 0.5)*vr
    if phi <= 0: return 0
    d = max(CAM_HEIGHT_M/math.tan(phi), 0.5)
    return math.atan((y2-y1)/fh*vr) * math.sqrt(CAM_HEIGHT_M**2+d**2)


# ── 색상 히스토그램 ───────────────────────────────────────────────────────────

def extract_color_hist(crop):
    h = crop.shape[0]; lower = crop[h//2:, :]
    if lower.shape[0] < 5: return None
    hsv  = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[18,8],[0,180,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def color_sim(h1, h2):
    if h1 is None or h2 is None: return 0.5
    return float(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32),
                                  cv2.HISTCMP_CORREL))


# ── 구역 ──────────────────────────────────────────────────────────────────────

def load_zones(cam=None):
    """
    cam 지정 시: 해당 카메라에 polygon이 설정된 구역만 반환
    cam=None   : 어느 카메라에든 polygon이 있는 구역 반환 (표시용)
    skip으로 표시된 구역은 제외.
    """
    if not ZONES_FILE.exists():
        print("[경고] zones.json 없음"); return []
    with open(ZONES_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    result = []
    for z in raw:
        polys = z.get("polygons", {})
        if cam:
            pts = polys.get(cam, [])
        else:
            # cam 미지정: 첫 번째 유효한 폴리곤 사용
            pts = next((v for v in polys.values()
                        if isinstance(v, list) and len(v) >= 3), [])
        if isinstance(pts, list) and len(pts) >= 3:
            result.append({**z, "polygon": np.array(pts, dtype=np.int32)})
    return result


def point_in_zone(cx, cy, zones):
    return [z["name"] for z in zones
            if cv2.pointPolygonTest(z["polygon"], (float(cx),float(cy)), False) >= 0]


# ── 트랙 데이터 ───────────────────────────────────────────────────────────────

class TrackData:
    def __init__(self, fps, timeline=None):
        self.fps=fps; self.timeline=timeline; self.tracks={}

    def get(self, tid):
        if tid not in self.tracks:
            self.tracks[tid] = {
                "tid":tid, "embeddings":[], "color_hists":[], "body_props":[],
                "gender":None, "gender_conf":0.0, "is_adult":None,
                "zone_entry":{}, "zone_dwell":defaultdict(float),
                "purchased":False, "purchased_items":[],
                "purchase_start":None, "purchase_duration":0.0,
                "first_dt":None, "last_dt":None,
            }
        return self.tracks[tid]

    def update_time(self, tid, fi):
        dt = frame_to_dt(fi, self.timeline)
        if dt is None: return
        t = self.get(tid)
        if t["first_dt"] is None: t["first_dt"] = dt
        t["last_dt"] = dt

    def add_features(self, tid, emb, hist, prop):
        t = self.get(tid)
        if emb  is not None: t["embeddings"].append(emb)
        if hist is not None: t["color_hists"].append(hist)
        if prop > 0:         t["body_props"].append(prop)

    def mean_embed(self, tid):
        e = self.get(tid)["embeddings"]
        if not e: return None
        m=np.mean(e,axis=0); n=np.linalg.norm(m)
        return m/n if n>0 else m

    def mean_color(self, tid):
        h = self.get(tid)["color_hists"]
        return np.mean(h,axis=0).astype(np.float32) if h else None

    def mean_body(self, tid):
        p = self.get(tid)["body_props"]
        return float(np.mean(p)) if p else 0.0

    def enter_zone(self, tid, z, fi): self.get(tid)["zone_entry"].setdefault(z, fi)

    def leave_zone(self, tid, z, fi):
        t = self.get(tid)
        if z in t["zone_entry"]:
            dur = (fi - t["zone_entry"].pop(z)) / self.fps
            if dur >= DWELL_MIN_SEC: t["zone_dwell"][z] += dur

    def flush_zones(self, tid, fi):
        for z in list(self.get(tid)["zone_entry"]): self.leave_zone(tid, z, fi)

    def update_attr(self, tid, g, conf, is_adult):
        t = self.get(tid)
        if g and conf > t["gender_conf"]:
            t["gender"]=g; t["gender_conf"]=conf
        if is_adult is not None and t["is_adult"] is None:
            t["is_adult"] = is_adult

    def mark_purchase(self, tid): self.get(tid)["purchased"] = True


# ── 시각화 (display=True 일 때만) ─────────────────────────────────────────────

def draw_overlay(frame, zones, td, cur_boxes, cur_zm, gid_map,
                 fi, fps, total, cam, is_purchase):
    vis=frame.copy(); h,w=vis.shape[:2]; ov=vis.copy()
    for i,z in enumerate(zones):
        col=ZONE_COLORS[i%len(ZONE_COLORS)]
        cv2.fillPoly(ov,[z["polygon"]],col)
        cv2.polylines(vis,[z["polygon"]],True,col,2)
        M=cv2.moments(z["polygon"])
        if M["m00"]:
            cx,cy=int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])
            cv2.putText(vis,z["name"],(cx-40,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    cv2.addWeighted(ov,0.2,vis,0.8,0,vis)
    for tid,(x1,y1,x2,y2) in cur_boxes.items():
        t=td.get(tid); g=t["gender"]or"?"; p=t["purchased"]
        a=("성인"if t["is_adult"]else"미성년")if t["is_adult"]is not None else"?"
        gid=gid_map.get(tid,f"L{tid}")
        col=C_BUY if p else(C_MALE if g=="M"else C_FEMALE if g=="F"else C_UNK)
        cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
        lbl=f"{gid} {g}/{a}"+(" [구매]"if p else"")
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        cv2.rectangle(vis,(x1,y1-th-6),(x1+tw+4,y1),col,-1)
        cv2.putText(vis,lbl,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        iz=cur_zm.get(tid,set())
        if iz:
            ds=" ".join(f"{z}:{(fi-t['zone_entry'].get(z,fi))/fps:.0f}s"for z in iz)
            cv2.putText(vis,ds,(x1,y2+15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,0),1)
    dt_s=frame_to_dt(fi,td.timeline); dt_s=dt_s.strftime("%H:%M:%S")if dt_s else"--:--:--"
    pct=fi/total*100 if total else 0
    hud=[f"{cam}  {dt_s}  [{pct:.1f}%  {fi}/{total}]",
         f"현재: {len(cur_boxes)}명  누적: {len(td.tracks)}명"]
    if is_purchase: hud.append(f"구매: {sum(1 for d in td.tracks.values() if d['purchased'])}건")
    for i,line in enumerate(hud):
        cv2.rectangle(vis,(0,i*22),(w,i*22+22),(0,0,0),-1)
        cv2.putText(vis,line,(6,i*22+16),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,180),1)
    return cv2.resize(vis,(int(w*DISPLAY_SCALE),int(h*DISPLAY_SCALE)))


# ── 단일 카메라 처리 ──────────────────────────────────────────────────────────

def process_camera(video_path, zones, reid, timeline,
                   is_purchase_cam=False, gid_map=None,
                   display=True, frame_skip=FRAME_SKIP,
                   biz_window=(None, None),
                   checkout_det=None):
    """
    Returns: (TrackData, metrics_dict)
    metrics_dict keys: cam, tracks, frames_proc, frames_total, proc_fps,
                       gender_rate, gender_dist, gender_conf_mean, adult_rate
    """
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam   = video_path.stem
    td    = TrackData(fps, timeline)
    model = YOLO(YOLO_MODEL)
    # model.model.half() 제거 — model.track(half=True)와 이중 변환 시 dtype 충돌 발생
    prev_z  = defaultdict(set)
    gid_map = gid_map or {}
    step    = frame_skip + 1
    t0      = time.time()
    fi=0; np_=0   # fi=실제 프레임 번호, np_=처리된 프레임 수

    if display:
        cv2.namedWindow(f"분석: {cam}", cv2.WINDOW_NORMAL)

    biz_start, biz_end = biz_window
    skipped_biz = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if fi % step != 0:
            fi += 1; continue

        # 영업 시간 창 밖이면 스킵 (ch3/ch6 적용)
        if biz_start is not None and timeline is not None:
            dt = frame_to_dt(fi, timeline)
            if dt is not None and (dt < biz_start or dt > biz_end):
                fi += 1; skipped_biz += 1; continue

        np_ += 1

        res = model.track(frame, persist=True, classes=[0],
                          conf=CONF, iou=IOU, device=DEVICE, verbose=False,
                          half=(USE_FP16 and DEVICE=="cuda"))
        cur_boxes={}; cur_tids=set(); cur_zm={}

        if res[0].boxes.id is not None:
            ids  = res[0].boxes.id.cpu().numpy().astype(int)
            xyxy = res[0].boxes.xyxy.cpu().numpy().astype(int)
            kall = (res[0].keypoints.data.cpu().numpy()
                    if res[0].keypoints is not None else None)

            for idx,(tid,box) in enumerate(zip(ids,xyxy)):
                x1,y1,x2,y2=box; cx,cy=(x1+x2)//2, y2
                cur_tids.add(tid); cur_boxes[tid]=(x1,y1,x2,y2)
                td.update_time(tid, fi)

                nz=set(point_in_zone(cx,cy,zones))
                cur_zm[tid]=nz
                for z in nz-prev_z[tid]: td.enter_zone(tid,z,fi)
                for z in prev_z[tid]-nz: td.leave_zone(tid,z,fi)
                prev_z[tid]=nz

                if is_purchase_cam and zones:
                    t=td.get(tid)
                    checkout_nz = {z for z in nz if z in ("계산대",)}
                    if checkout_nz:
                        t["purchase_start"]=t["purchase_start"] or fi
                    elif t["purchase_start"] is not None:
                        dur=(fi-t["purchase_start"])/fps
                        t["purchase_duration"]+=dur
                        if dur>=PURCHASE_DWELL_SEC: td.mark_purchase(tid)
                        t["purchase_start"]=None

                if np_ % EMBED_EVERY == 0:
                    x1c,y1c=max(0,x1),max(0,y1)
                    x2c,y2c=min(frame.shape[1],x2),min(frame.shape[0],y2)
                    crop=frame[y1c:y2c,x1c:x2c]
                    td.add_features(tid, reid.extract(crop),
                                    extract_color_hist(crop),
                                    body_proportion(y1,y2,fh))

                if np_ % ATTR_EVERY == 0:
                    kpts=(kall[idx] if kall is not None and idx<len(kall) else None)
                    g,c=gender_from_keypoints(kpts)
                    is_adult=(adult_from_keypoints(kpts,fh) if kpts is not None
                              else estimate_adult(y1,y2,fh))
                    td.update_attr(tid,g,c,is_adult)

        # ── 계산대 상품 감지 (ch1 전용) ──────────────────────────────────────
        if checkout_det is not None and cur_tids:
            result = checkout_det.update(frame, fi)
            if result:
                # 스캐너 구역에 가장 가까이 있는 트랙에 구매 기록
                scanner_tids = [tid for tid in cur_tids
                                if "스캐너" in cur_zm.get(tid, set())]
                target_tids  = scanner_tids if scanner_tids else list(cur_tids)
                for tid in target_tids:
                    t = td.get(tid)
                    t["purchased"] = True
                    t["purchased_items"].append(result)
                    print(f"    [구매] 트랙{tid} → {result['product']} "
                          f"/ {result['zone_name']} / {result['price']}원 "
                          f"(유사도={result['similarity']})")

        for tid in list(prev_z):
            if tid not in cur_tids:
                td.flush_zones(tid,fi); del prev_z[tid]

        if display:
            cv2.imshow(f"분석: {cam}",
                       draw_overlay(frame,zones,td,cur_boxes,cur_zm,
                                    gid_map,fi,fps,total,cam,is_purchase_cam))
            if cv2.waitKey(1)&0xFF in(ord('q'),ord('Q')):
                print("  [건너뜀]"); break

        fi+=1
        if np_%300==0:
            elapsed=time.time()-t0
            print(f"    {fi}/{total} ({fi/total*100:.1f}%) "
                  f"| {np_/elapsed:.1f}fps | 추적={len(td.tracks)}명")

    for tid in list(prev_z): td.flush_zones(tid,fi)
    cap.release()
    if display: cv2.destroyWindow(f"분석: {cam}")

    # ── 메트릭 집계 ──────────────────────────────────────────────────────────
    elapsed=max(time.time()-t0, 0.001)
    n=len(td.tracks)
    gc={"M":0,"F":0,"?":0}; gconfs=[]; adult_n=0
    for t in td.tracks.values():
        g=t["gender"]or"?"
        gc[g if g in gc else"?"]+=1
        if g!="?" and t.get("gender_conf",0)>0: gconfs.append(t["gender_conf"])
        if t["is_adult"] is True: adult_n+=1

    gdet=gc["M"]+gc["F"]
    m={
        "cam":              cam,
        "tracks":           n,
        "frames_proc":      np_,
        "frames_total":     total,
        "frames_biz_skip":  skipped_biz,
        "proc_fps":         round(np_/elapsed,1),
        "gender_rate":      round(gdet/n,3) if n>0 else 0.0,
        "gender_dist":      gc,
        "gender_conf_mean": round(float(np.mean(gconfs)),3) if gconfs else 0.0,
        "adult_rate":       round(adult_n/n,3) if n>0 else 0.0,
    }
    if n>0:
        biz_skip_pct = skipped_biz/(total//step+1)*100 if total>0 else 0
        print(f"  → {n}명 추적 | {m['proc_fps']}fps | "
              f"영업창 외 스킵={biz_skip_pct:.1f}% | "
              f"성별감지={gdet/n*100:.1f}% (avg신뢰={m['gender_conf_mean']:.2f}) | "
              f"성인={adult_n/n*100:.1f}%")
    else:
        print(f"  → 0명 추적")
    return td, m


# ── 5-신호 Re-ID 점수 ────────────────────────────────────────────────────────
# ch3↔ch6 직접 매칭 제거. ch1 트랙이 유일한 anchor.

WINDOW_BUFFER_SEC = 120   # ch1 창 앞뒤 여유 (이동 시간 + 클럭 오차 보정)

def reid_score(tda, ta, tdb, tb):
    """
    외모(0.35) + 색상(0.30) + 체형(0.20) + 시간(0.15)
    ch1↔store 매칭이므로 동선 순서 제약(W_SEQ) 제거
    """
    ea=tda.mean_embed(ta); eb=tdb.mean_embed(tb)
    ap=(max(0,1-float(cdist([ea],[eb],"cosine")[0,0]))
        if ea is not None and eb is not None else 0.5)
    cs=(color_sim(tda.mean_color(ta),tdb.mean_color(tb))+1)/2
    pa=tda.mean_body(ta); pb=tdb.mean_body(tb)
    bs=(max(0,1-abs(pa-pb)/max(pa,pb)) if pa>0 and pb>0 else 0.5)
    da=tda.get(ta)["first_dt"]; db=tdb.get(tb)["first_dt"]
    if da and db:
        diff=abs((da-db).total_seconds())
        ts=max(0,1-diff/TIME_WINDOW_SEC)
    else: ts=0.5
    return 0.35*ap + 0.30*cs + 0.20*bs + 0.15*ts


# ── ch1 기준 방문 창으로 ch3/ch6 매칭 ────────────────────────────────────────

def match_by_ch1_window(ch1_td, store_tds):
    """
    ch1 트랙별 방문 창 [first_dt - buf, last_dt + buf] 안의
    ch3/ch6 트랙을 Re-ID 점수로 매칭.

    특징:
    - ch3↔ch6 직접 비교 없음 (ch1이 유일한 기준)
    - 한 ch1 트랙에 ch3/ch6 트랙 여러 개 허용 (왔다갔다)
    - 하나의 store 트랙은 가장 높은 점수의 ch1 트랙에만 배정

    Returns:
        {ch1_tid: [(cam, store_tid, score), ...]}
    """
    buf = timedelta(seconds=WINDOW_BUFFER_SEC)

    # ── 1단계: 시간 창 안의 후보 수집 + Re-ID 점수 계산 ──
    # store_track → 경쟁하는 ch1 후보들 {(cam,s_tid): [(ch1_tid, score)]}
    competition = defaultdict(list)

    for ch1_tid, ch1_t in ch1_td.tracks.items():
        w_s = ch1_t["first_dt"]; w_e = ch1_t["last_dt"]
        if w_s is None: continue
        w_s -= buf; w_e = (w_e or w_s) + buf

        for cam, td in store_tds.items():
            for s_tid, s_t in td.tracks.items():
                t_s = s_t["first_dt"]; t_e = s_t["last_dt"] or t_s
                if t_s is None: continue
                # 시간 창 겹침 확인
                if t_e < w_s or t_s > w_e: continue
                score = reid_score(ch1_td, ch1_tid, td, s_tid)
                if score >= REID_THRESHOLD:
                    competition[(cam, s_tid)].append((ch1_tid, score))

    # ── 2단계: 각 store 트랙을 최고 점수의 ch1 트랙 하나에만 배정 ──
    matches = defaultdict(list)   # ch1_tid → [(cam, s_tid, score)]
    for (cam, s_tid), rivals in competition.items():
        best_ch1_tid, best_score = max(rivals, key=lambda x: x[1])
        matches[best_ch1_tid].append((cam, s_tid, best_score))

    n_store = sum(len(v) for v in matches.values())
    n_ch1   = sum(1 for v in matches.values() if v)
    print(f"  ch1 창 기반 매칭: ch1 {n_ch1}/{len(ch1_td.tracks)}명 연결 "
          f"| store 트랙 {n_store}개 배정")

    # 매칭 상세 출력
    for ch1_tid, mlist in matches.items():
        ch1_t = ch1_td.tracks[ch1_tid]
        w_s = ch1_t["first_dt"]; w_e = ch1_t["last_dt"]
        cams_str = ", ".join(f"{c}[{round(s,2)}]" for c,_,s in mlist)
        print(f"    V{ch1_tid:04d} [{w_s:%H:%M}~{w_e:%H:%M}] ← {cams_str or '없음'}")

    return dict(matches)


# ── 결과 통합 ─────────────────────────────────────────────────────────────────

def build_visit_data(ch1_td, matches, store_tds):
    """
    ch1 트랙 = 방문자 ID 기준.
    매칭된 ch3/ch6 트랙의 zone_dwell을 모두 합산.
    성별/나이는 신뢰도 최고 트랙에서 채택.
    """
    visits = {}
    counter = 1

    for ch1_tid, ch1_t in ch1_td.tracks.items():
        gvid    = f"V{counter:04d}"; counter += 1
        matched = matches.get(ch1_tid, [])

        # 방문 카메라 목록 (ch1 + 매칭된 카메라, 순서 유지)
        cams_seen = ["ch1"]
        for cam, _, _ in matched:
            if cam not in cams_seen: cams_seen.append(cam)

        # 성별/나이: ch1 + 매칭 트랙 중 신뢰도 최고
        best_g    = ch1_t["gender"]
        best_conf = ch1_t.get("gender_conf", 0)
        best_adult= ch1_t["is_adult"]
        for cam, s_tid, _ in matched:
            t = store_tds[cam].tracks[s_tid]
            if t.get("gender_conf", 0) > best_conf:
                best_g = t["gender"]; best_conf = t["gender_conf"]
            if best_adult is None and t["is_adult"] is not None:
                best_adult = t["is_adult"]

        # 구역 체류시간: 매칭된 ch3/ch6 트랙 모두 합산
        zones = {}
        for cam, s_tid, _ in matched:
            for zone, dur in store_tds[cam].tracks[s_tid]["zone_dwell"].items():
                zones[zone] = zones.get(zone, 0) + dur

        entry = ch1_t["first_dt"]
        exit_ = ch1_t["last_dt"]

        # 구매 상품 목록 (스캐너 감지 결과)
        p_items = ch1_t.get("purchased_items", [])
        # 신뢰도 내림차순 정렬, 중복 상품 제거
        seen_products = set()
        deduped = []
        for item in sorted(p_items, key=lambda x: x.get("similarity", 0), reverse=True):
            name = item.get("product", "")
            if name not in seen_products:
                deduped.append(item)
                seen_products.add(name)

        visits[gvid] = {
            "visit_id":       gvid,
            "cameras":        cams_seen,
            "gender":         best_g,
            "is_adult":       best_adult,
            "zones":          zones,
            "purchased":      ch1_t["purchased"],
            "purchased_items": deduped,   # [{product, zone_name, zone_id, price, similarity}]
            "entry_time":     entry.strftime("%H:%M:%S") if entry else None,
            "exit_time":      exit_.strftime("%H:%M:%S") if exit_ else None,
            "timeband":       get_timeband(entry.strftime("%H:%M:%S") if entry else None),
            "matched_store_tracks": len(matched),
        }

    matched_any = sum(1 for v in visits.values() if len(v["cameras"]) > 1)
    print(f"  방문자 총 {len(visits)}명 | 매대 방문 추적: {matched_any}명")
    return visits


# ── 진입점 (run_pipeline.py 에서 호출) ───────────────────────────────────────

def ch1_clip_window(day):
    """ch1 클립 파일명에서 영업 시간 창 추출 → (start_dt, end_dt)"""
    clips = sorted((BASE_DIR / "ch1" / day).glob("*.mp4"))
    if not clips:
        return None, None
    from merge import parse_clip
    times = [parse_clip(c) for c in clips]
    times = [(s, e) for r in times if r for _, s, e in [r]]
    if not times: return None, None
    return min(s for s, _ in times), max(e for _, e in times)


def run_analysis(day="10", display=False):
    """
    Returns: (visits_dict, summary_metrics)
    구조:
      1. ch1 먼저 처리 (anchor: 구매 감지 + 방문 창 정의)
      2. ch3/ch6 처리 — ch1 영업 창 밖 프레임 스킵
      3. ch1 개인별 창으로 ch3/ch6 트랙 매칭 (왔다갔다 허용)
      4. 방문 데이터 통합
    """
    print(f"Device: {DEVICE} "
          f"({'FP16 ' if USE_FP16 else ''}"
          f"{torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")
    zones = load_zones()
    print(f"구역 {len(zones)}개 로드 | FRAME_SKIP={FRAME_SKIP} (처리율 1/{FRAME_SKIP+1})\n")

    print("Re-ID 모델 로딩...")
    reid = ReIDExtractor()
    print("완료\n")

    all_metrics = []

    # ── Step 1: ch1 처리 (구매 감지 포함) ───────────────────────────────────
    ch1_td   = TrackData(fps=15)
    ch1_path = MOSAIC_DIR / f"ch1_{day}.mp4"
    if not ch1_path.exists():
        print("[경고] ch1 영상 없음 — 구매/방문 창 정의 불가")
        return None, {"error": "ch1 없음"}

    zones_ch1 = load_zones("ch1")
    tl = load_timeline("ch1", day)
    print(f"[처리 1/3] ch1_{day}.mp4 | 타임라인={'있음' if tl else '없음'} | 구역={len(zones_ch1)}개")

    # 스캐너 구역 폴리곤이 설정된 경우 상품 감지기 생성
    checkout_det = None
    scanner_zone = next((z for z in zones_ch1
                         if z.get("id") == "scanner"
                         and z.get("polygon") is not None
                         and len(z.get("polygon", [])) >= 3), None)
    if scanner_zone is not None:
        from checkout_detector import ProductMatcher, ScannerDetector
        matcher      = ProductMatcher()
        checkout_det = ScannerDetector(scanner_zone["polygon"], matcher)
        print("  [스캐너] 상품 감지기 활성")
    else:
        print("  [경고] 스캐너 구역 폴리곤 미설정 → 상품 감지 비활성")

    ch1_td, m1 = process_camera(ch1_path, zones_ch1, reid, tl,
                                 is_purchase_cam=True, display=display,
                                 checkout_det=checkout_det)
    all_metrics.append(m1)

    # ch1 클립에서 영업 시간 창 추출
    biz_start, biz_end = ch1_clip_window(day)
    if biz_start:
        print(f"\n  ch1 영업 창: {biz_start:%H:%M:%S} ~ {biz_end:%H:%M:%S}")

    # ── Step 2: ch3/ch6 처리 (카메라별 zones, ch1 영업 창 밖 스킵) ──────────
    store_tds = {}
    for i, cam in enumerate(["ch3", "ch6"], start=2):
        path = MOSAIC_DIR / f"{cam}_{day}.mp4"
        if not path.exists():
            print(f"\n[SKIP] {path.name}"); continue
        zones_cam = load_zones(cam)
        tl = load_timeline(cam, day)
        print(f"\n[처리 {i}/3] {path.name} | 타임라인={'있음' if tl else '없음'} | 구역={len(zones_cam)}개")
        if biz_start:
            print(f"  영업 창 밖 프레임 스킵: {biz_start:%H:%M} 이전 / {biz_end:%H:%M} 이후")
        td, m = process_camera(path, zones_cam, reid, tl, display=display,
                               biz_window=(biz_start, biz_end))
        store_tds[cam] = td; all_metrics.append(m)

    if not store_tds:
        print("[경고] ch3/ch6 없음 — 매대 체류 분석 불가")

    # ── Step 3: ch1 개인별 방문 창으로 ch3/ch6 매칭 ──────────────────────────
    print("\n[Re-ID] ch1 방문 창 기반 매칭...")
    matches = match_by_ch1_window(ch1_td, store_tds) if store_tds else {}

    reid_m = {
        "ch1_tracks":  len(ch1_td.tracks),
        "ch1_matched": sum(1 for v in matches.values() if v),
        "store_tracks": {c: len(td.tracks) for c, td in store_tds.items()},
    }

    # ── Step 4: 방문 데이터 통합 ─────────────────────────────────────────────
    visits = build_visit_data(ch1_td, matches, store_tds)
    with open(RESULT_FILE,"w",encoding="utf-8") as f:
        json.dump(visits,f,ensure_ascii=False,indent=2,default=str)

    purchased=sum(1 for v in visits.values() if v["purchased"])
    print(f"\n완료: {len(visits)}명 | 구매자 {purchased}명 → {RESULT_FILE}")
    return visits, {"per_cam":all_metrics,"reid":reid_m,
                    "total_visitors":len(visits),"purchasers":purchased}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--day",       default="10")
    p.add_argument("--display",   action="store_true", default=True)
    p.add_argument("--no-display",dest="display",action="store_false")
    a=p.parse_args()
    run_analysis(day=a.day, display=a.display)

if __name__=="__main__":
    main()
