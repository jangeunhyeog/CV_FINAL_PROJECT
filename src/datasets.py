"""데이터셋 로더 — vpr-datasets-downloader의 표준 `@` 파일명 포맷을 파싱한다.

파일명 포맷(필드 구분자 '@'):
  @ UTM_east @ UTM_north @ UTM_zone_num @ UTM_zone_letter @ lat @ lon @
  pano_id @ tile_num @ heading @ pitch @ roll @ height @ timestamp @ note @ ext

정답 판정(positives)은 여기서만 결정한다 — UTM 25m 임계값(주행 데이터셋 표준).
평가 함수(eval_recall)는 이 기준을 모른다(프로토콜을 한 곳에 고정).
"""
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_POS_THRESH_M = 25.0


def parse_utm(path: str):
    """파일명에서 (UTM_east, UTM_north) 추출. 실패 시 None."""
    name = os.path.basename(path)
    parts = name.split("@")
    # parts[0]은 보통 빈 문자열(선행 @), parts[1]=east, parts[2]=north
    try:
        return float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        return None


def list_split(dataset_dir: str, split: str = "test",
               db_sub: str = "database", q_sub: str = "queries"):
    """datasets/<name>/images/<split>/{db_sub,q_sub}의 정렬된 파일 목록 반환.

    SVOX 등은 db_sub='gallery', q_sub='queries_night' 처럼 지정한다.
    """
    base = os.path.join(dataset_dir, "images", split)
    db = sorted(glob.glob(os.path.join(base, db_sub, "*")))
    q = sorted(glob.glob(os.path.join(base, q_sub, "*")))
    if not db or not q:
        raise SystemExit(f"{db_sub}/{q_sub} 비어있음: {base}")
    return db, q


def utm_array(paths):
    """경로 목록 → (N,2) UTM 배열. 파싱 실패 항목은 NaN."""
    out = np.full((len(paths), 2), np.nan, dtype=np.float64)
    for i, p in enumerate(paths):
        u = parse_utm(p)
        if u is not None:
            out[i] = u
    return out


def build_positives(q_utm: np.ndarray, db_utm: np.ndarray,
                    thresh_m: float = DEFAULT_POS_THRESH_M):
    """각 쿼리의 정답 DB 인덱스 집합. UTM 거리 <= thresh_m."""
    tree = cKDTree(db_utm)
    neigh = tree.query_ball_point(q_utm, r=thresh_m)
    return [set(idx) for idx in neigh]


def load_geo_dataset(dataset_dir: str, split: str = "test",
                     thresh_m: float = DEFAULT_POS_THRESH_M,
                     db_sub: str = "database", q_sub: str = "queries"):
    """UTM 기반 데이터셋(St Lucia 등) 로딩. queries는 정렬=주행 순서(모듈 A 전제)."""
    db, q = list_split(dataset_dir, split, db_sub, q_sub)
    db_utm, q_utm = utm_array(db), utm_array(q)
    positives = build_positives(q_utm, db_utm, thresh_m)
    return {
        "db_paths": db, "q_paths": q,
        "db_utm": db_utm, "q_utm": q_utm,
        "positives": positives,
        "thresh_m": thresh_m,
    }


if __name__ == "__main__":
    import sys
    d = load_geo_dataset(sys.argv[1])
    n_pos = sum(len(p) for p in d["positives"])
    empty = sum(1 for p in d["positives"] if not p)
    print(f"db={len(d['db_paths'])} q={len(d['q_paths'])} "
          f"avg_positives={n_pos/len(d['positives']):.2f} "
          f"queries_with_no_gt={empty} thresh={d['thresh_m']}m")
