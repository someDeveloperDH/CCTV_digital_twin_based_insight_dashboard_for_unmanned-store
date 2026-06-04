"""
파이프라인 검증 스크립트
정답 CSV 3개 vs 파이프라인 출력 CSV 3개 비교
- 헝가리안 알고리즘으로 에이전트 1:1 매칭
- 상세 오류 분석 + validation_report.csv 저장
실행: python validate.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def read_csv(path):
    """utf-8-sig → cp949 → euc-kr 순으로 인코딩 자동 감지"""
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"인코딩 감지 실패: {path}")

BASE_DIR = Path(__file__).parent

# ── 정답 파일 (ground_truth/ice_*.csv) ───────────────────────────────────────
GT_DIR       = BASE_DIR / "ground_truth"
GT_PURCHASES = GT_DIR / "ice_batch_purchases.csv"
GT_CLASS_KPI = GT_DIR / "ice_class_kpi.csv"
GT_ZONE_KPI  = GT_DIR / "ice_zone_kpi.csv"

# ── 파이프라인 출력 파일 (cctv1/ice_*.csv) ────────────────────────────────────
PR_PURCHASES = BASE_DIR / "ice_batch_purchases.csv"
PR_CLASS_KPI = BASE_DIR / "ice_class_kpi.csv"
PR_ZONE_KPI  = BASE_DIR / "ice_zone_kpi.csv"

# ── 리포트 출력 ───────────────────────────────────────────────────────────────
REPORT_FILE  = BASE_DIR / "validation_report.csv"


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def mape(true_vals, pred_vals):
    """평균절대백분율오차 (0으로 나누는 경우 제외)"""
    errors = []
    for t, p in zip(true_vals, pred_vals):
        if t != 0:
            errors.append(abs(t - p) / abs(t) * 100)
    return round(np.mean(errors), 2) if errors else 0.0


def mae(true_vals, pred_vals):
    return round(np.mean([abs(t - p) for t, p in zip(true_vals, pred_vals)]), 2)


def f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    return round(2 * prec * rec / (prec + rec) * 100, 1) if (prec + rec) > 0 else 0.0


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def print_section(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


# ── 에이전트별 집계 ───────────────────────────────────────────────────────────

def aggregate_by_agent(df):
    """
    ice_batch_purchases 형식 → 에이전트별 요약 dict 반환
    {agent_id: {cls, cls_name, zones, products, total_price, purchased, dwell}}
    """
    agents = {}
    for aid, grp in df.groupby("에이전트ID"):
        zones    = set(grp[grp["구역ID"] != "none"]["구역ID"].tolist())
        products = set(grp[grp["상품명"] != "none"]["상품명"].tolist())
        purchased = bool(products)
        total_price = grp["가격"].sum()
        total_dwell = grp.drop_duplicates("구역ID")["체류시간"].sum()
        cls  = grp["고객분류"].iloc[0]
        cls_name = grp["고객분류명"].iloc[0]
        agents[aid] = {
            "id":          aid,
            "cls":         cls,
            "cls_name":    cls_name,
            "zones":       zones,
            "products":    products,
            "purchased":   purchased,
            "total_price": total_price,
            "total_dwell": total_dwell,
        }
    return agents


# ── 1:1 매칭 (헝가리안 알고리즘) ─────────────────────────────────────────────

def build_similarity_matrix(gt_agents, pr_agents):
    """
    GT × PR 유사도 행렬 (높을수록 유사)
    항목별 가중치:
      분류 일치     0.30
      구역 Jaccard  0.30
      구매 일치     0.25
      체류시간 근접 0.15
    """
    gt_ids = list(gt_agents.keys())
    pr_ids = list(pr_agents.keys())
    mat    = np.zeros((len(gt_ids), len(pr_ids)))

    max_dwell = max(
        max((a["total_dwell"] for a in gt_agents.values()), default=1),
        max((a["total_dwell"] for a in pr_agents.values()), default=1),
        1
    )

    for i, gid in enumerate(gt_ids):
        g = gt_agents[gid]
        for j, pid in enumerate(pr_ids):
            p = pr_agents[pid]

            cls_score   = 1.0 if g["cls"] == p["cls"] else 0.0
            zone_score  = jaccard(g["zones"], p["zones"])
            purch_score = 1.0 if g["purchased"] == p["purchased"] else 0.0
            dwell_diff  = abs(g["total_dwell"] - p["total_dwell"]) / max_dwell
            dwell_score = max(0, 1 - dwell_diff)

            mat[i, j] = (0.30 * cls_score +
                         0.30 * zone_score +
                         0.25 * purch_score +
                         0.15 * dwell_score)
    return gt_ids, pr_ids, mat


def match_agents(gt_agents, pr_agents):
    """헝가리안 알고리즘으로 GT ↔ PR 최적 1:1 매칭"""
    gt_ids, pr_ids, mat = build_similarity_matrix(gt_agents, pr_agents)

    # scipy는 최소화, 유사도는 최대화 → 부호 반전
    row_ind, col_ind = linear_sum_assignment(-mat)

    matched   = []
    unmatched_gt = set(gt_ids)
    unmatched_pr = set(pr_ids)

    for r, c in zip(row_ind, col_ind):
        score = mat[r, c]
        matched.append((gt_ids[r], pr_ids[c], round(score, 3)))
        unmatched_gt.discard(gt_ids[r])
        unmatched_pr.discard(pr_ids[c])

    return matched, list(unmatched_gt), list(unmatched_pr)


# ── 상세 비교 ─────────────────────────────────────────────────────────────────

def compare_agents(matched, gt_agents, pr_agents):
    """매칭된 쌍별 필드 비교 → 상세 결과 rows"""
    rows = []
    for gt_id, pr_id, sim in matched:
        g = gt_agents[gt_id]
        p = pr_agents[pr_id]

        cls_ok    = g["cls"] == p["cls"]
        purch_ok  = g["purchased"] == p["purchased"]
        zone_jacc = round(jaccard(g["zones"], p["zones"]), 3)
        dwell_err = round(abs(g["total_dwell"] - p["total_dwell"]), 1)
        dwell_pct = round(abs(g["total_dwell"] - p["total_dwell"])
                          / g["total_dwell"] * 100, 1) if g["total_dwell"] else "-"

        # 구역 오류 상세
        missed_zones = g["zones"] - p["zones"]   # GT에 있는데 예측 못함
        extra_zones  = p["zones"] - g["zones"]   # 예측했는데 GT에 없음

        error_notes = []
        if not cls_ok:
            error_notes.append(f"분류오류({g['cls']}→{p['cls']})")
        if not purch_ok:
            error_notes.append("구매" + ("미감지" if g["purchased"] else "오감지"))
        if missed_zones:
            error_notes.append(f"구역누락:{','.join(missed_zones)}")
        if extra_zones:
            error_notes.append(f"구역오감지:{','.join(extra_zones)}")

        rows.append({
            "GT_ID":       gt_id,
            "PR_ID":       pr_id,
            "매칭점수":    sim,
            "분류_GT":     g["cls"],
            "분류_PR":     p["cls"],
            "분류일치":    "✅" if cls_ok else "❌",
            "구매_GT":     "O" if g["purchased"] else "X",
            "구매_PR":     "O" if p["purchased"] else "X",
            "구매일치":    "✅" if purch_ok else "❌",
            "구역_Jacc":   zone_jacc,
            "체류_GT(s)":  g["total_dwell"],
            "체류_PR(s)":  p["total_dwell"],
            "체류오차(s)": dwell_err,
            "체류오차(%)": dwell_pct,
            "오류내용":    " | ".join(error_notes) if error_notes else "-",
        })
    return rows


# ── KPI 비교 ──────────────────────────────────────────────────────────────────

def compare_class_kpi(gt_df, pr_df):
    gt = gt_df.set_index("분류")
    pr = pr_df.set_index("분류") if "분류" in pr_df.columns else pr_df.set_index(pr_df.columns[0])
    rows = []
    for cls in gt.index:
        if cls not in pr.index:
            rows.append({"분류": cls, "항목": "전체", "GT": "-", "PR": "-", "오차(%)": "미감지"})
            continue
        for col in ["방문객수", "구매고객수", "구매전환율(%)", "총매출", "평균구매액"]:
            if col not in gt.columns:
                continue
            gv = gt.loc[cls, col]
            pv = pr.loc[cls, col] if col in pr.columns else 0
            err = round(abs(gv - pv) / gv * 100, 1) if gv != 0 else "-"
            rows.append({"분류": cls, "항목": col,
                         "GT": gv, "PR": pv, "오차(%)": err})
    return pd.DataFrame(rows)


def compare_zone_kpi(gt_df, pr_df):
    gt = gt_df.set_index("구역")
    pr = pr_df.set_index("구역") if "구역" in pr_df.columns else pr_df.set_index(pr_df.columns[0])
    rows = []
    for zone in gt.index:
        if zone not in pr.index:
            rows.append({"구역": zone, "항목": "전체", "GT": "-", "PR": "-", "오차(%)": "미감지"})
            continue
        for col in ["방문객수", "구매고객수", "구매전환율(%)", "평균체류시간(초)", "총매출", "비효율지수"]:
            if col not in gt.columns:
                continue
            gv = gt.loc[zone, col]
            pv = pr.loc[zone, col] if col in pr.columns else 0
            err = round(abs(gv - pv) / gv * 100, 1) if gv != 0 else "-"
            rows.append({"구역": zone, "항목": col,
                         "GT": gv, "PR": pv, "오차(%)": err})
    return pd.DataFrame(rows)


# ── 종합 지표 ─────────────────────────────────────────────────────────────────

def summary_metrics(detail_rows, unmatched_gt, unmatched_pr):
    total     = len(detail_rows)
    cls_acc   = sum(1 for r in detail_rows if r["분류일치"] == "✅") / total * 100 if total else 0
    purch_acc = sum(1 for r in detail_rows if r["구매일치"] == "✅") / total * 100 if total else 0
    avg_jacc  = np.mean([r["구역_Jacc"] for r in detail_rows]) * 100 if total else 0

    # 구매 F1
    tp = sum(1 for r in detail_rows if r["구매_GT"] == "O" and r["구매_PR"] == "O")
    fp = sum(1 for r in detail_rows if r["구매_GT"] == "X" and r["구매_PR"] == "O")
    fn = sum(1 for r in detail_rows if r["구매_GT"] == "O" and r["구매_PR"] == "X")

    dwell_errs = [r["체류오차(%)"] for r in detail_rows if isinstance(r["체류오차(%)"], float)]

    return {
        "총 GT 에이전트":    len(detail_rows) + len(unmatched_gt),
        "총 PR 에이전트":    len(detail_rows) + len(unmatched_pr),
        "매칭 성공":         len(detail_rows),
        "GT 미매칭":         len(unmatched_gt),
        "PR 미매칭":         len(unmatched_pr),
        "분류 정확도(%)":    round(cls_acc, 1),
        "구매 감지 F1(%)":   f1(tp, fp, fn),
        "구역 Jaccard(%)":   round(avg_jacc, 1),
        "체류시간 MAPE(%)":  round(np.mean(dwell_errs), 1) if dwell_errs else "-",
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # 파일 존재 확인
    missing = [f for f in [GT_PURCHASES, GT_CLASS_KPI, GT_ZONE_KPI,
                             PR_PURCHASES, PR_CLASS_KPI, PR_ZONE_KPI]
               if not f.exists()]
    if missing:
        print("다음 파일이 없습니다:")
        for f in missing:
            tag = "[정답]" if f.parent == GT_DIR else "[예측]"
            print(f"  {tag} {f.name}")
        print("\n정답 파일 위치: ground_truth/ice_batch_purchases.csv 등")
        return

    gt_pur = read_csv(GT_PURCHASES)
    gt_cls = read_csv(GT_CLASS_KPI)
    gt_zon = read_csv(GT_ZONE_KPI)
    pr_pur = read_csv(PR_PURCHASES)
    pr_cls = read_csv(PR_CLASS_KPI)
    pr_zon = read_csv(PR_ZONE_KPI)

    # ── 에이전트 집계 ──
    gt_agents = aggregate_by_agent(gt_pur)
    pr_agents = aggregate_by_agent(pr_pur)

    # ── 1:1 매칭 ──
    print("헝가리안 알고리즘으로 에이전트 매칭 중...")
    matched, unmatched_gt, unmatched_pr = match_agents(gt_agents, pr_agents)
    print(f"  매칭 완료: {len(matched)}쌍 / GT 미매칭 {len(unmatched_gt)}명 / PR 미매칭 {len(unmatched_pr)}명")

    # ── 상세 비교 ──
    detail_rows = compare_agents(matched, gt_agents, pr_agents)
    metrics     = summary_metrics(detail_rows, unmatched_gt, unmatched_pr)

    # ── 출력: 종합 지표 ──
    print_section("[ 종합 지표 ]")
    print(tabulate([[k, v] for k, v in metrics.items()],
                   headers=["항목", "값"], tablefmt="rounded_outline"))

    # ── 출력: 에이전트별 상세 비교 ──
    print_section("[ 에이전트 1:1 매칭 상세 ]")
    show_cols = ["GT_ID","PR_ID","매칭점수","분류일치","구매일치",
                 "구역_Jacc","체류오차(s)","체류오차(%)","오류내용"]
    print(tabulate([{c: r[c] for c in show_cols} for r in detail_rows],
                   headers="keys", tablefmt="rounded_outline", showindex=False))

    # ── 출력: 오류 에이전트만 따로 ──
    error_rows = [r for r in detail_rows if r["오류내용"] != "-"]
    if error_rows:
        print_section(f"[ 오류 에이전트 상세 ] ({len(error_rows)}명)")
        print(tabulate([{c: r[c] for c in show_cols} for r in error_rows],
                       headers="keys", tablefmt="rounded_outline", showindex=False))

    # ── 출력: 미매칭 에이전트 ──
    if unmatched_gt or unmatched_pr:
        print_section("[ 미매칭 에이전트 ]")
        if unmatched_gt:
            print(f"  GT에만 있음 (파이프라인이 못 잡음): {unmatched_gt}")
        if unmatched_pr:
            print(f"  PR에만 있음 (오감지): {unmatched_pr}")

    # ── 출력: 분류 KPI 비교 ──
    df_cls_cmp = compare_class_kpi(gt_cls, pr_cls)
    print_section("[ 고객 분류 KPI 비교 ]")
    print(tabulate(df_cls_cmp, headers="keys",
                   tablefmt="rounded_outline", showindex=False))

    # ── 출력: 구역 KPI 비교 ──
    df_zon_cmp = compare_zone_kpi(gt_zon, pr_zon)
    print_section("[ 구역 KPI 비교 ]")
    print(tabulate(df_zon_cmp, headers="keys",
                   tablefmt="rounded_outline", showindex=False))

    # ── 오류 유형 빈도 분석 ──
    print_section("[ 오류 유형 분석 ]")
    error_types = defaultdict(int)
    for r in error_rows:
        for note in r["오류내용"].split(" | "):
            if note != "-":
                # 오류 유형 단순화
                if "분류오류" in note:   error_types["분류 오류"] += 1
                elif "미감지" in note:   error_types["구매 미감지"] += 1
                elif "오감지" in note and "구역" not in note: error_types["구매 오감지"] += 1
                elif "구역누락" in note: error_types["구역 누락"] += 1
                elif "구역오감지" in note: error_types["구역 오감지"] += 1
    if error_types:
        print(tabulate([[k, v, f"{v/len(detail_rows)*100:.1f}%"]
                        for k, v in sorted(error_types.items(), key=lambda x: -x[1])],
                       headers=["오류 유형", "건수", "비율"],
                       tablefmt="rounded_outline"))
    else:
        print("  오류 없음")

    # ── CSV 저장 ──
    report_rows = []
    for r in detail_rows:
        report_rows.append({**r, "구분": "매칭"})
    for gid in unmatched_gt:
        g = gt_agents[gid]
        report_rows.append({
            "GT_ID": gid, "PR_ID": "없음", "매칭점수": 0,
            "분류_GT": g["cls"], "분류_PR": "-", "분류일치": "❌",
            "구매_GT": "O" if g["purchased"] else "X", "구매_PR": "-", "구매일치": "❌",
            "구역_Jacc": 0, "체류_GT(s)": g["total_dwell"], "체류_PR(s)": 0,
            "체류오차(s)": g["total_dwell"], "체류오차(%)": 100,
            "오류내용": "파이프라인 미감지", "구분": "GT미매칭"
        })
    for pid in unmatched_pr:
        p = pr_agents[pid]
        report_rows.append({
            "GT_ID": "없음", "PR_ID": pid, "매칭점수": 0,
            "분류_GT": "-", "분류_PR": p["cls"], "분류일치": "❌",
            "구매_GT": "-", "구매_PR": "O" if p["purchased"] else "X", "구매일치": "❌",
            "구역_Jacc": 0, "체류_GT(s)": 0, "체류_PR(s)": p["total_dwell"],
            "체류오차(s)": p["total_dwell"], "체류오차(%)": 100,
            "오류내용": "오감지", "구분": "PR미매칭"
        })

    pd.DataFrame(report_rows).to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n리포트 저장: {REPORT_FILE}")


def run_validation():
    """
    run_pipeline.py 에서 호출하는 진입점.
    Returns: {"cls_acc", "purchase_f1", "zone_jaccard", "dwell_mape", "overall_score"}
    or None on failure.
    """
    missing = [f for f in [GT_PURCHASES, GT_CLASS_KPI, GT_ZONE_KPI,
                             PR_PURCHASES, PR_CLASS_KPI, PR_ZONE_KPI]
               if not f.exists()]
    if missing:
        return None

    gt_pur = read_csv(GT_PURCHASES)
    pr_pur = read_csv(PR_PURCHASES)

    gt_agents = aggregate_by_agent(gt_pur)
    pr_agents = aggregate_by_agent(pr_pur)
    matched, unmatched_gt, unmatched_pr = match_agents(gt_agents, pr_agents)
    detail_rows = compare_agents(matched, gt_agents, pr_agents)
    m = summary_metrics(detail_rows, unmatched_gt, unmatched_pr)

    total = len(detail_rows)
    cls_acc     = m["분류 정확도(%)"] / 100 if total else 0
    purchase_f1 = float(str(m["구매 감지 F1(%)"]).replace("%","")) / 100 if total else 0
    zone_jacc   = m["구역 Jaccard(%)"] / 100 if total else 0
    dwell_mape  = float(str(m["체류시간 MAPE(%)"]).replace("-","100")) / 100

    overall = round(0.35*cls_acc + 0.30*purchase_f1 + 0.25*zone_jacc
                    + 0.10*max(0, 1-dwell_mape), 4)
    return {
        "cls_acc":      round(cls_acc,    4),
        "purchase_f1":  round(purchase_f1,4),
        "zone_jaccard": round(zone_jacc,  4),
        "dwell_mape":   round(dwell_mape, 4),
        "overall_score":overall,
        "matched":      len(matched),
        "unmatched_gt": len(unmatched_gt),
        "unmatched_pr": len(unmatched_pr),
    }


if __name__ == "__main__":
    main()
