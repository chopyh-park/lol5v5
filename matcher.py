# -*- coding: utf-8 -*-
"""
LoL 5v5 팀 매칭 스크립트 (v7.6)
================================

이 버전은 v7.5를 기반으로 다음과 같은 개선을 적용한다.

* **그래프 복원**: 라인별 Δ 그래프를 v6 스타일로 되돌렸다. 가운데 `|` 문자의
  위치가 모든 행에서 고정되며, 왼쪽과 오른쪽 영역은 동일한 폭을 갖는다.
  Δ = (Team B 점수 − Team A 점수)에 따라 블럭(`█`)이 가운데에서
  바깥 방향으로 채워지며, 반대편은 공백으로 유지된다.
* **승률 가중치 조정**: cost 계산에서 승률 편차에 대한 가중치를 대폭 높였다.
  40~60% 승률 구간을 우대하는 기본 구조는 유지하되, 60%를 넘는 구간에
  훨씬 더 가파른 페널티를 적용하고, 승률 항의 가중치를 15배로 상향했다.
  이를 통해 승률이 60% 이상인 매치는 라인과 분산이 아무리 좋아도
  상위 후보에서 밀려나도록 했다.
* **나머지 구조 유지**: 선형 MMR, λ·w 계산, DFS 탐색, 윤영/우재 라인 고정,
  금지 매치업, NOT_SAME_TEAM 등 v7.5의 설계는 그대로 유지한다.

"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import math
import sys
import re
import json
import os

# 버전 문자열
VERSION = "v7.6"
WIN_PROB_LIMIT = 0.60  # 강제 승률 캡 (A 또는 B의 승률이 60%를 넘으면 제외)
# 탐색 시 유지할 상위 후보 개수(메모리·시간 절약을 위해 전체 전수를 저장하지 않는다)
TOP_N_RESULTS = 200
STATE_LIMIT = 20000  # DFS 노드 확장 상한 (시간 폭주 방지)

def _slug_ver(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

STATE_PATH = f"/tmp/tsr_state_{_slug_ver(VERSION)}.json"

ROLES = ['Top','Jg','Mid','Adc','Sup']
TIER_ORDER = ['Iron','Bronze','Silver','Gold','Platinum','Emerald','Diamond','Master','Grandmaster','Challenger']

###############################################################################
# optional YAML rule loader
try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

def load_rules(path: str = 'rules.yaml') -> Dict[str, any]:
    if yaml is None or not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            return data or {}
    except Exception:
        return {}

_rules_data = load_rules()
_default_player_pair_target = {('혜원','Jg'):'Bronze II 0LP', ('윤주','Jg'):'Bronze II 0LP'}
_default_role_overrides     = {('윤영','Jg'):'Iron I 70LP'}
_default_forbid_matchups    = {('윤영','우재'), ('용훈','혜원')}
_default_not_same           = {('고래','윤영')}

PLAYER_PAIR_TARGET_TIERS: Dict[Tuple[str,str], str] = _default_player_pair_target.copy()
ROLE_SCORE_OVERRIDES: Dict[Tuple[str,str], str]     = _default_role_overrides.copy()
FORBID_MATCHUPS: Set[Tuple[str,str]] = _default_forbid_matchups.copy()
NOT_SAME_TEAM: Set[Tuple[str,str]]    = _default_not_same.copy()

# override with rules.yaml if present
if _rules_data:
    if 'player_pair_target_tiers' in _rules_data:
        try:
            PLAYER_PAIR_TARGET_TIERS = {(tpl[0], tpl[1]): tpl[2] for tpl in _rules_data['player_pair_target_tiers']}
        except Exception:
            pass
    if 'role_score_overrides' in _rules_data:
        try:
            ROLE_SCORE_OVERRIDES = {(tpl[0], tpl[1]): tpl[2] for tpl in _rules_data['role_score_overrides']}
        except Exception:
            pass
    if 'forbid_matchups' in _rules_data:
        try:
            FORBID_MATCHUPS = {tuple(item) for item in _rules_data['forbid_matchups']}
        except Exception:
            pass
    if 'not_same_team' in _rules_data:
        try:
            NOT_SAME_TEAM = {tuple(item) for item in _rules_data['not_same_team']}
        except Exception:
            pass

###############################################################################
# 티어 및 MMR 함수
TIER_INDEX = {
    'Iron':0, 'Bronze':1, 'Silver':2, 'Gold':3,
    'Platinum':4, 'Emerald':5, 'Diamond':6,
    'Master':7, 'Grandmaster':8, 'Challenger':9,
}

def division_index(div: int) -> int:
    """디비전(IV=4~I=1)을 0~3 인덱스로 변환한다."""
    return max(0, min(3, 4 - div))

def tier_to_mmr(tier: str, div: Optional[int], lp: int) -> float:
    """선형 MMR 매핑: Iron~Diamond는 티어*400 + 디비전*100 + LP, Master 이상은 LP만 반영."""
    idx_t = TIER_INDEX[tier]
    lp_val = lp
    if tier in ('Master','Grandmaster','Challenger'):
        return idx_t * 400 + lp_val
    d_idx = division_index(div if div is not None else 2)
    return idx_t * 400 + d_idx * 100 + lp_val

def tier_level_scalar(tier: str) -> float:
    """티어를 0~2 범위의 스칼라로 누르는 함수."""
    idx = TIER_INDEX[tier]
    if idx <= 5:
        return idx / 5.0
    elif idx <= 8:
        return 1.0 + (idx - 5) / 3.0
    else:
        return 2.0

def w_from_tier(tier: str) -> float:
    """티어에 따른 w 값 계산. 저티어는 0, 중간은 1, 상위는 0.2까지 감소."""
    s = tier_level_scalar(tier)
    if s <= 1.0:
        return s
    w = 1.0 - 0.8 * (s - 1.0)
    return max(0.2, w)

@dataclass
class Player:
    name: str
    pref_role: str
    tier_text: str
    specialist: bool

def _norm_role(r: str) -> str:
    r = r.strip().lower()
    if r in ('top','탑'): return 'Top'
    if r in ('jg','jungle','정글'): return 'Jg'
    if r in ('mid','미드'): return 'Mid'
    if r in ('adc','bot','원딜','바텀'): return 'Adc'
    if r in ('sup','support','서폿','서포터'): return 'Sup'
    raise ValueError(f'알 수 없는 라인: {r}')

def _norm_spec(s: str) -> bool:
    s = s.strip().lower()
    return ('전문' in s) or ('spec' in s)

def parse_players(text: str) -> List[Player]:
    out: List[Player] = []
    for i, line in enumerate(text.splitlines(),1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            raise ValueError(f'{i}행 형식 오류: 이름, 선호라인, 티어, 전문형/범용형')
        out.append(Player(parts[0], _norm_role(parts[1]), parts[2], _norm_spec(parts[3])))
    return out

def parse_tier_text(t: str) -> Tuple[str, int, int]:
    """티어 문자열을 파싱해 (티어, 디비전, LP)를 반환한다."""
    s = t.strip().replace('Lp','LP').replace('lp','LP')
    s_lower = s.lower()
    tier = 'Silver'
    # 긴 이름을 우선 매칭
    for tier_name in sorted(TIER_ORDER, key=lambda x: -len(x)):
        pattern = r'\b' + re.escape(tier_name.lower()) + r'\b'
        if re.search(pattern, s_lower):
            tier = tier_name
            break
    token = None
    for tok in ['IV','III','II','I','4','3','2','1']:
        if re.search(r'\b' + re.escape(tok) + r'\b', s):
            token = tok
            break
    if token is None:
        token = 'II'
    div_map = {'IV':4,'III':3,'II':2,'I':1,'4':4,'3':3,'2':2,'1':1}
    div = div_map.get(token, 2)
    lp = 0
    for word in s.split():
        if word.endswith('LP'):
            try:
                lp = int(word[:-2])
            except ValueError:
                pass
    return tier, div, lp

###############################################################################
# λ 테이블 (전문형과 범용형 전환 효율)
LAMBDA_GEN: Dict[str, Dict[str, float]] = {
    'Top':{'Top':1.00,'Jg':0.75,'Mid':0.85,'Adc':0.80,'Sup':0.78},
    'Jg' :{'Top':0.80,'Jg':1.00,'Mid':0.85,'Adc':0.76,'Sup':0.73},
    'Mid':{'Top':0.84,'Jg':0.80,'Mid':1.00,'Adc':0.86,'Sup':0.85},
    'Adc':{'Top':0.85,'Jg':0.72,'Mid':0.85,'Adc':1.00,'Sup':0.88},
    'Sup':{'Top':0.70,'Jg':0.58,'Mid':0.80,'Adc':0.73,'Sup':1.00},
}
LAMBDA_SPEC: Dict[str, Dict[str, float]] = {
    'Top':{'Top':1.00,'Jg':0.60,'Mid':0.70,'Adc':0.68,'Sup':0.66},
    'Jg' :{'Top':0.70,'Jg':1.00,'Mid':0.60,'Adc':0.63,'Sup':0.61},
    'Mid':{'Top':0.70,'Jg':0.60,'Mid':1.00,'Adc':0.67,'Sup':0.70},
    'Adc':{'Top':0.87,'Jg':0.56,'Mid':0.60,'Adc':1.00,'Sup':0.60},
    'Sup':{'Top':0.50,'Jg':0.40,'Mid':0.70,'Adc':0.62,'Sup':1.00},
}

def player_role_score(player: Player, assigned_role: str) -> float:
    """플레이어가 특정 라인에 배정될 때의 MMR 점수."""
    tier, div, lp = parse_tier_text(player.tier_text)
    key = (player.name, assigned_role)
    if key in ROLE_SCORE_OVERRIDES:
        t, d, l = parse_tier_text(ROLE_SCORE_OVERRIDES[key])
        return tier_to_mmr(t, d, l)
    elif key in PLAYER_PAIR_TARGET_TIERS:
        t, d, l = parse_tier_text(PLAYER_PAIR_TARGET_TIERS[key])
        return tier_to_mmr(t, d, l)
    base_mmr = tier_to_mmr(tier, div, lp)
    if assigned_role == player.pref_role:
        return base_mmr
    # 오프롤일 때 w·λ 페널티 적용
    w = w_from_tier(tier)
    lam_table = LAMBDA_SPEC if player.specialist else LAMBDA_GEN
    lam = lam_table[player.pref_role][assigned_role]
    factor = (1.0 - w) + w * lam
    return base_mmr * factor

def compute_logistic_scale(assignA, assignB) -> float:
    """팀 평균 티어에 따라 로지스틱 스케일을 결정한다."""
    indices: List[int] = []
    for pl, _, _ in assignA + assignB:
        tier, div, lp = parse_tier_text(pl.tier_text)
        indices.append(TIER_INDEX[tier])
    avg_idx = sum(indices) / len(indices)
    if avg_idx >= 8.0:
        return 180.0
    elif avg_idx >= 7.0:
        return 220.0
    elif avg_idx >= 6.0:
        return 260.0
    elif avg_idx >= 5.0:
        return 310.0
    else:
        return 400.0

def winprob_logistic(assignA, assignB, scale: Optional[float] = None) -> float:
    """로지스틱 함수로 Team A 승률을 계산한다."""
    sumA = sum(score for _,_,score in assignA)
    sumB = sum(score for _,_,score in assignB)
    diff = sumA - sumB
    if scale is None:
        scale = compute_logistic_scale(assignA, assignB)
    return 1.0 / (1.0 + math.exp(-diff / scale))

###############################################################################
# 상태 저장/로드
def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(st: dict) -> None:
    try:
        with open(STATE_PATH, 'w') as f:
            json.dump(st, f)
    except Exception:
        pass

###############################################################################
# 출력 유틸리티
def _render_table(headers, rows):
    widths = []
    for i, h in enumerate(headers):
        col = [len(str(r[i])) for r in rows] if rows else [0]
        widths.append(max(len(str(h)), max(col)))
    top = "┌" + "┬".join("─"*(w+2) for w in widths) + "┐"
    mid = "├" + "┼".join("─"*(w+2) for w in widths) + "┤"
    bot = "└" + "┴".join("─"*(w+2) for w in widths) + "┘"
    print(top)
    print("│ " + " │ ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " │")
    print(mid)
    for r in rows:
        print("│ " + " │ ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " │")
    print(bot)

def _diff_bar(d: float, max_abs: float = 700.0, max_blocks: int = 6) -> str:
    """
    Δ = B - A. 가운데 '|'를 기준으로 좌우 영역을 고정하고, 강한 쪽으로만 블럭을 채운다.
    - d < 0 (A 우세): 왼쪽 영역의 오른쪽부터 █가 채워진다.
    - d > 0 (B 우세): 오른쪽 영역의 왼쪽부터 █가 채워진다.
    - d = 0: 양쪽 모두 공백.
    전체 길이는 max_blocks*2 + 1 고정이다.
    """
    # 비율 계산
    ratio = min(1.0, max(-1.0, d / max_abs))
    blocks = int(abs(ratio) * max_blocks + 0.5)
    if d < 0:
        left = ("█" * blocks).rjust(max_blocks)
        right = " " * max_blocks
    elif d > 0:
        left = " " * max_blocks
        right = ("█" * blocks).ljust(max_blocks)
    else:
        left = " " * max_blocks
        right = " " * max_blocks
    return f"{left}|{right}"

def print_side_by_side(assignA, assignB):
    rows = []
    for r in ROLES:
        a = next(item for item in assignA if item[1] == r)
        b = next(item for item in assignB if item[1] == r)
        sa = int(round(a[2]))
        sb = int(round(b[2]))
        astr = f"{a[0].name} ({sa}, {'전문' if a[0].specialist else '범용'})"
        bstr = f"{b[0].name} ({sb}, {'전문' if b[0].specialist else '범용'})"
        delta = sb - sa  # B - A
        rows.append([r, astr, _diff_bar(delta), f"{delta:+.0f}", bstr])
    _render_table(["Role","Team A","Diff","Δ","Team B"], rows)

def lane_deltas(assignA, assignB) -> Dict[str, float]:
    d: Dict[str,float] = {}
    for r in ROLES:
        sa = next(score for _, rr, score in assignA if rr == r)
        sb = next(score for _, rr, score in assignB if rr == r)
        d[r] = sb - sa
    return d

def print_result(assignA, assignB, P, tag=None, cost: Optional[float]=None):
    print(f"# === LoL 5v5 수동 매칭 — 완전탐색 [{VERSION}] ===")
    if tag:
        print(tag)
    print(f"예상 승률: Team A {P*100:.1f}%  /  Team B {(1-P)*100:.1f}%")
    if cost is not None:
        print(f"[cost={cost:.4f}]")
    print("[Lane-by-Lane — A | Δ | B]")
    print_side_by_side(assignA, assignB)
    deltas = lane_deltas(assignA, assignB)
    ssum = ", ".join([f"{r}:{int(round(deltas[r])):+d}" for r in ROLES])
    print(f"Δ summary  →  {ssum}")

###############################################################################
# DFS 탐색
SLOTS: List[Tuple[str,str]] = [
    ('A','Top'),('A','Jg'),('A','Mid'),('A','Adc'),('A','Sup'),
    ('B','Top'),('B','Jg'),('B','Mid'),('B','Adc'),('B','Sup'),
]

def search_assignments(players: List[Player], allowed_roles: Dict[str, Set[str]], *, top_n: int = TOP_N_RESULTS) -> List[Tuple[List[Tuple[Player,str,float]], List[Tuple[Player,str,float]], float, float, float, float, float]]:
    """
    DFS로 10개 슬롯에 플레이어를 배치하고, cost를 계산한다.
    반환값은 (assignA, assignB, P, term_win, term_lane, term_var, cost) 리스트이다.
    """
    import heapq
    heap: List[Tuple[float, float, int, List[Tuple[Player,str,float]], List[Tuple[Player,str,float]], float, float, float, float]] = []
    counter = 0
    visits = 0
    n = len(players)
    forbid = FORBID_MATCHUPS
    not_same = NOT_SAME_TEAM

    def dfs(remaining: List[Tuple[str,str]], used: List[bool], assignA: List[Tuple[Player,str,float]], assignB: List[Tuple[Player,str,float]],
            namesA: Set[str], namesB: Set[str], lane_mapA: Dict[str,str], lane_mapB: Dict[str,str]):
        nonlocal visits
        visits += 1
        if visits > STATE_LIMIT:
            return
        if not remaining:
            P = winprob_logistic(assignA, assignB)
            p_major = max(P, 1.0 - P)
            gap = p_major - 0.5
            if gap <= 0.1:
                term_win = gap / 0.1
            else:
                term_win = 1.0 + 10.0 * (gap - 0.1) / 0.4
            deltas = lane_deltas(assignA, assignB)
            lane_quad = sum((abs(val) / 200.0) ** 2 for val in deltas.values())
            lane_over = sum(max(0.0, abs(val) - 200) / 200.0 for val in deltas.values())
            term_lane = 5.0 * lane_quad + 1.5 * lane_over
            mmrsA = [s for _,_,s in assignA]
            mmrsB = [s for _,_,s in assignB]
            meanA = sum(mmrsA) / len(mmrsA)
            meanB = sum(mmrsB) / len(mmrsB)
            varA  = sum((s - meanA)**2 for s in mmrsA) / len(mmrsA)
            varB  = sum((s - meanB)**2 for s in mmrsB) / len(mmrsB)
            term_var = (varA + varB) / (400.0**2)
            cost = 12.0 * term_win + term_lane + 0.6 * term_var
            nonlocal counter
            counter += 1
            heapq.heappush(heap, (-cost, -abs(P - 0.5), counter, assignA.copy(), assignB.copy(), P, term_win, term_lane, term_var))
            if len(heap) > top_n:
                heapq.heappop(heap)
            return

        best_slot: Optional[Tuple[str,str]] = None
        best_candidates: List[int] = []
        best_len = 10**9
        for team, lane in remaining:
            candidates: List[int] = []
            for i in range(n):
                if used[i]:
                    continue
                pl = players[i]
                if pl.name in allowed_roles and lane not in allowed_roles[pl.name]:
                    continue
                if team == 'A':
                    violation = False
                    for a,b in not_same:
                        if (pl.name == a and b in namesA) or (pl.name == b and a in namesA):
                            violation = True; break
                    if violation:
                        continue
                else:
                    violation = False
                    for a,b in not_same:
                        if (pl.name == a and b in namesB) or (pl.name == b and a in namesB):
                            violation = True; break
                    if violation:
                        continue
                if team == 'A' and lane in lane_mapB:
                    opponent = lane_mapB[lane]
                    if (pl.name, opponent) in forbid or (opponent, pl.name) in forbid:
                        continue
                if team == 'B' and lane in lane_mapA:
                    opponent = lane_mapA[lane]
                    if (pl.name, opponent) in forbid or (opponent, pl.name) in forbid:
                        continue
                candidates.append(i)
            if not candidates:
                return
            if len(candidates) < best_len:
                best_len = len(candidates)
                best_slot = (team, lane)
                best_candidates = candidates

        assert best_slot is not None
        team, lane = best_slot
        next_remaining = [s for s in remaining if s != best_slot]
        for i in best_candidates:
            used[i] = True
            pl = players[i]
            mmr = player_role_score(pl, lane)
            if team == 'A':
                assignA.append((pl, lane, mmr))
                namesA.add(pl.name)
                lane_mapA[lane] = pl.name
                dfs(next_remaining, used, assignA, assignB, namesA, namesB, lane_mapA, lane_mapB)
                lane_mapA.pop(lane)
                namesA.remove(pl.name)
                assignA.pop()
            else:
                assignB.append((pl, lane, mmr))
                namesB.add(pl.name)
                lane_mapB[lane] = pl.name
                dfs(next_remaining, used, assignA, assignB, namesA, namesB, lane_mapA, lane_mapB)
                lane_mapB.pop(lane)
                namesB.remove(pl.name)
                assignB.pop()
            used[i] = False

    used = [False] * n
    dfs(SLOTS.copy(), used, [], [], set(), set(), {}, {})
    # 비용 오름차순으로 정렬해 반환
    out: List[Tuple[List[Tuple[Player,str,float]], List[Tuple[Player,str,float]], float, float, float, float, float]] = []
    while heap:
        neg_cost, neg_bal, _, assignA, assignB, P, term_win, term_lane, term_var = heapq.heappop(heap)
        cost = -neg_cost
        out.append((assignA, assignB, P, term_win, term_lane, term_var, cost))
    return list(sorted(out, key=lambda x: (x[6], abs(x[2] - 0.5))))

###############################################################################
# 샘플 플레이어 데이터
PLAYERS_TEXT = """
혜원, Sup, Platinum 2 30LP, 전문형
윤영, Top, Iron 3 30LP, 전문형
윤주, Sup, Platinum 2 30LP, 전문형
기운, Adc, Gold 3 20LP, 전문형
흥석, Top, Diamond 3 30Lp, 범용형
세훈, Mid, Gold 1 20LP, 범용형
현빈, Adc, Gold 4 20LP, 전문형
용훈, Adc, Diamond 2 20LP, 범용형
성한, Sup, Platinum 4 92Lp, 전문형
우재, Mid, Bronze 2 40LP, 전문형
""".strip()

def parse_players_text(text: str) -> List[Player]:
    out = []
    for i, line in enumerate(text.splitlines(),1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            raise ValueError(f'{i}행 형식 오류')
        out.append(Player(parts[0], _norm_role(parts[1]), parts[2], _norm_spec(parts[3])))
    return out

###############################################################################
# 메인 실행
if __name__ == '__main__':
    players = parse_players_text(PLAYERS_TEXT)
    if len(players) != 10:
        print(f"[경고] 현재 인원 {len(players)}명 (10명 권장)", file=sys.stderr)

    # 윤영을 Top/Jg 양쪽으로 검색해 더 균형 잡힌 후보를 확보한다.
    forced_roles = ['Top', 'Jg']
    combined_results: List[Tuple[List[Tuple[Player,str,float]], List[Tuple[Player,str,float]], float, float, float, float, float, str]] = []
    for forced_role in forced_roles:
        allowed_roles: Dict[str, Set[str]] = {
            '우재': {'Mid'},
            '윤영': {forced_role},
            '혜원': {'Top','Jg','Mid','Sup'},
        }
        raw_results = search_assignments(players, allowed_roles)
        # 승률 캡을 적용해 60% 초과 매치를 강제로 제거한다.
        capped_results = []
        for assignA, assignB, P, term_win, term_lane, term_var, cost in raw_results:
            p_major = max(P, 1.0 - P)
            if p_major > WIN_PROB_LIMIT:
                continue
            capped_results.append((assignA, assignB, P, term_win, term_lane, term_var, cost))
        effective_results = capped_results if capped_results else raw_results
        for item in effective_results:
            combined_results.append((*item, forced_role))

    # 대칭 매치 제거 (강제 역할 조합이 달라도 동일 인원 구성이면 중복 제거)
    seen_keys: Set[Tuple[Tuple[str,...], Tuple[str,...]]] = set()
    filtered: List[Tuple[List[Tuple[Player,str,float]], List[Tuple[Player,str,float]], float, float, float, float, float, str]] = []
    for assignA, assignB, P, term_win, term_lane, term_var, cost, forced_role in combined_results:
        namesA = tuple(sorted(p.name for p,_,_ in assignA))
        namesB = tuple(sorted(p.name for p,_,_ in assignB))
        key = tuple(sorted([namesA, namesB]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        filtered.append((assignA, assignB, P, term_win, term_lane, term_var, cost, forced_role))

    results_sorted = sorted(filtered, key=lambda x: (x[6], abs(x[2] - 0.5)))
    for idx, (assignA, assignB, P, term_win, term_lane, term_var, cost, forced_role) in enumerate(results_sorted[:3], 1):
        tag = f"(윤영 강제 역할: {forced_role})  (우재 Mid 고정 적용)"
        print(f"\n## 후보 {idx}")
        print_result(assignA, assignB, P, tag=tag, cost=cost)
        print(f"[승률={P:.3f}, term_win={term_win:.3f}, term_lane={term_lane:.3f}, term_var={term_var:.3f}, cost={cost:.3f}]")
