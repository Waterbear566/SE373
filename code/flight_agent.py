"""BTVN#3 · SE373 — Agent đặt vé máy bay với LangChain + LangGraph + harness.

Nội dung file này
  1. Ràng buộc là dữ liệu (Constraints) và tiêu chí hoàn thành kiểm bằng code (completion_ok)
  2. Backend giả (FlightWorld) + 5 tool mockup (LangChain @tool) + các kịch bản kiểm thử
  3. Các lớp harness: validator, constraint guard, kiểm quyền, loop/stall detector,
     ngân sách, grounding check, bàn giao (handoff)
  4. SimModel: model giả lập có kiểm soát (BaseChatModel) để chạy offline, lặp lại được.
     Có API key thì dùng model thật: đặt SE373_MODEL rồi chạy với --real.
  5. FlightAgent: ba mẫu thiết kế dựng bằng LangGraph StateGraph
        react   : model → act → check → model ...
        plan    : plan → execute → check → ... → finalize → verify  (không lập lại kế hoạch)
        hybrid  : như plan nhưng observation đổi đáng kể thì replan

Chạy thử:
    python flight_agent.py --pattern hybrid --scenario sold_out
    python flight_agent.py --pattern react  --scenario need_approval --approve
    python flight_agent.py --pattern react  --scenario forget_constraint --no-harness
"""
import argparse
import json
import os
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date as _date, timedelta
from typing import Annotated, Callable, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

GOAL = "Đặt giúp tôi một vé SGN → DAD sáng 07/10/2026, giá dưới 2 triệu."


def fmt(p: int) -> str:
    return f"{p:,}".replace(",", ".")


# ════════════════════════════════════════════════════════════════════
# 1. RÀNG BUỘC LÀ DỮ LIỆU + TIÊU CHÍ HOÀN THÀNH
# ════════════════════════════════════════════════════════════════════
class Constraints(BaseModel):
    """Yêu cầu của người dùng, ghi thành dữ liệu để kiểm lại bằng code (không nằm trong trí nhớ model)."""
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"
    max_price: int = 2_000_000
    auto_limit: int = 1_500_000      # vượt mức này phải có người duyệt (lớp kiểm quyền)

    def checks(self, f: dict, price: Optional[int] = None) -> dict:
        p = f["price"] if price is None else price
        return {
            "route": f["origin"] == self.origin and f["destination"] == self.destination,
            "date": f["date"] == self.date,
            "time": f["depart"] < self.depart_before,
            "price": p <= self.max_price,
        }

    def violations(self, f: dict, price: Optional[int] = None, ignore_time: bool = False) -> list:
        return [k for k, ok in self.checks(f, price).items() if not ok and not (ignore_time and k == "time")]

    def is_ok(self, f: dict, price: Optional[int] = None, ignore_time: bool = False) -> bool:
        return not self.violations(f, price, ignore_time)


def completion_ok(b: Optional[dict], c: Constraints) -> tuple:
    """Tiêu chí hoàn thành: vị từ chạy bằng code, độc lập với lời model (slide: sensor computational)."""
    if not b:
        return False, ["chưa có booking"]
    bad = []
    if b.get("status") != "confirmed":
        bad.append("status != confirmed")
    if not b.get("paid"):
        bad.append("chưa thanh toán")
    bad += [f"vi phạm {k}" for k in c.violations(b)]
    return not bad, bad


# ════════════════════════════════════════════════════════════════════
# 2. BACKEND GIẢ + TOOL MOCKUP + KỊCH BẢN
# ════════════════════════════════════════════════════════════════════
def F(flight, depart, price, refundable=True, date="2026-10-07"):
    return {"flight": flight, "origin": "SGN", "destination": "DAD", "date": date,
            "depart": depart, "price": price, "refundable": refundable}


@dataclass
class Scenario:
    name: str
    desc: str
    flights: list
    seats: dict = field(default_factory=dict)             # chuyến → số ghế còn (mặc định 3)
    timeout: set = field(default_factory=set)             # chuyến lỗi timeout ở check_seat/book_seat
    price_at_check: dict = field(default_factory=dict)    # giá thật lúc check_seat khác giá lúc tìm
    legacy_errors: bool = False                           # tool cũ trả "not found" thay vì hint (slide: tool trả lỗi kém)
    flaws: set = field(default_factory=set)               # lỗi của MODEL, chỉ SimModel đọc


_BASE = [F("VN122", "08:10", 1_350_000), F("VJ604", "10:40", 1_450_000), F("QH118", "15:40", 1_640_000)]

SCENARIOS = {s.name: s for s in [
    Scenario("happy", "Đường thẳng: mọi thứ hoạt động", _BASE),
    Scenario("sold_out", "Chuyến rẻ nhất hết chỗ lúc check_seat", _BASE, seats={"VN122": 0}),
    Scenario("timeout", "check_seat/book_seat VN122 timeout, model cứ gọi lại", _BASE,
             timeout={"VN122"}, flaws={"loop_on_error"}),
    Scenario("price_changed", "Giá lúc check_seat vượt trần 2 triệu", _BASE, price_at_check={"VN122": 2_250_000}),
    Scenario("need_approval", "Mọi chuyến hợp lệ đều vượt hạn mức 1,5tr và không hoàn",
             [F("VN122", "08:10", 1_950_000, False), F("VJ604", "10:40", 1_980_000, False), F("QH118", "15:40", 1_640_000)]),
    Scenario("no_valid_flight", "Không có chuyến nào thoả đủ ràng buộc; model đi lang thang",
             [F("VN122", "08:10", 2_310_000), F("VJ604", "10:40", 2_080_000), F("QH118", "15:40", 1_640_000)],
             flaws={"wander_on_empty"}),
    Scenario("hallucinate", "Model báo 'đã đặt VN999...' khi chưa đặt gì", _BASE, flaws={"hallucinate_final"}),
    Scenario("bad_date", "Model gửi date='07/10'; tool cũ chỉ trả 'not found'", _BASE,
             legacy_errors=True, flaws={"bad_date", "loop_on_error"}),
    Scenario("forget_constraint", "Model quên ràng buộc giờ bay, chọn chuyến chiều rẻ nhất",
             [F("QH118", "15:40", 1_150_000), F("VN122", "08:10", 1_350_000), F("VJ604", "10:40", 1_450_000)],
             flaws={"ignore_time"}),
]}


class FlightWorld:
    """Hệ thống đặt vé giả. Trả JSON có status rõ ràng: ok / error / invalid_param."""

    def __init__(self, scenario: Scenario):
        self.s = scenario
        self.bookings: dict = {}
        self.payments: list = []
        self._codes = iter(["4XJ2", "7KQ9", "2MB5", "9ZP3", "5RT8", "3LN6"])

    def _flight(self, fid):
        return next((f for f in self.s.flights if f["flight"] == fid), None)

    def search_flights(self, origin, destination, date):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date)):
            if self.s.legacy_errors:
                return {"status": "error", "error": "not found"}
            return {"status": "invalid_param", "param": "date", "hint": "Dùng YYYY-MM-DD, ví dụ 2026-10-07"}
        fl = [f for f in self.s.flights if (f["origin"], f["destination"], f["date"]) == (origin, destination, date)]
        return {"status": "ok", "flights": [dict(f) for f in fl]}

    def check_seat(self, flight):
        f = self._flight(flight)
        if not f:
            return {"status": "error", "error": "unknown_flight"}
        if flight in self.s.timeout:
            return {"status": "error", "error": "timeout", "hint": "Dịch vụ ghế lỗi, thử lại sau"}
        n = self.s.seats.get(flight, 3)
        return {"status": "ok", "flight": flight, "seats": n,
                "available_seats": ["12A", "12B", "12C"][:n],
                "price": self.s.price_at_check.get(flight, f["price"])}

    def book_seat(self, flight, seat):
        f = self._flight(flight)
        if not f:
            return {"status": "error", "error": "unknown_flight"}
        if flight in self.s.timeout:
            return {"status": "error", "error": "timeout"}
        if self.s.seats.get(flight, 3) == 0:
            return {"status": "error", "error": "sold_out"}
        code = next(self._codes)
        price = self.s.price_at_check.get(flight, f["price"])
        self.bookings[code] = {**f, "code": code, "seat": seat, "price": price, "status": "held", "paid": False}
        return {"status": "ok", "booking_code": code, "booking_status": "held", "price": price}

    def pay(self, booking_code, method):
        b = self.bookings.get(booking_code)
        if not b:
            return {"status": "error", "error": "booking_not_found"}
        if b["paid"]:
            return {"status": "error", "error": "already_paid"}
        b["paid"], b["status"] = True, "confirmed"
        self.payments.append({"code": booking_code, "price": b["price"], "refundable": b["refundable"], "flight": b["flight"]})
        return {"status": "ok", "paid": True}

    def get_booking(self, booking_code):
        b = self.bookings.get(booking_code)
        return {"status": "ok", "booking": dict(b)} if b else {"status": "error", "error": "booking_not_found"}


def make_tools(world: FlightWorld) -> list:
    """5 tool mockup theo giao thức tool calling (tên + mô tả + schema tham số)."""
    def js(x):
        return json.dumps(x, ensure_ascii=False)

    @tool
    def search_flights(origin: str, destination: str, date: str) -> str:
        """Tìm chuyến bay. origin/destination là mã IATA (SGN, DAD); date dạng YYYY-MM-DD."""
        return js(world.search_flights(origin, destination, date))

    @tool
    def check_seat(flight: str) -> str:
        """Kiểm tra ghế trống và giá hiện hành của một chuyến, ví dụ flight='VN122'."""
        return js(world.check_seat(flight))

    @tool
    def book_seat(flight: str, seat: str) -> str:
        """Giữ chỗ một ghế (ví dụ seat='12A'). Trả về booking_code."""
        return js(world.book_seat(flight, seat))

    @tool
    def pay(booking_code: str, method: str) -> str:
        """Thanh toán booking_code bằng method (ví dụ 'corp_card')."""
        return js(world.pay(booking_code, method))

    @tool
    def get_booking(booking_code: str) -> str:
        """Đọc lại trạng thái booking (status, paid, price...)."""
        return js(world.get_booking(booking_code))

    return [search_flights, check_seat, book_seat, pay, get_booking]


# ════════════════════════════════════════════════════════════════════
# 3. CÁC LỚP HARNESS
# ════════════════════════════════════════════════════════════════════
class Term:
    DONE, BUDGET, LOOP, STALL, NEED_HUMAN = "DONE", "BUDGET", "LOOP", "STALL", "NEED_HUMAN"


IATA_RE = re.compile(r"[A-Z]{3}")
FLIGHT_RE = re.compile(r"[A-Z0-9]{2}\d{3,4}")
SEAT_RE = re.compile(r"\d{1,2}[A-F]")
CODE_RE = re.compile(r"[0-9A-Z]{4}")
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class Validator:
    """Chặn gọi tool không tồn tại và tham số sai định dạng TRƯỚC khi thực thi, trả hint để model sửa."""

    def __init__(self, names):
        self.names = set(names)

    def check(self, name, args, allow_placeholder=False) -> Optional[dict]:
        if name not in self.names:
            return {"status": "invalid_tool", "tool": name, "hint": f"Chỉ dùng: {sorted(self.names)}"}
        rules = {
            "search_flights": {"origin": IATA_RE, "destination": IATA_RE, "date": ISO_RE},
            "check_seat": {"flight": FLIGHT_RE},
            "book_seat": {"flight": FLIGHT_RE, "seat": SEAT_RE},
            "pay": {"booking_code": CODE_RE, "method": re.compile(r"corp_card|card|cash")},
            "get_booking": {"booking_code": CODE_RE},
        }[name]
        hints = {"date": "Dùng YYYY-MM-DD, ví dụ 2026-10-07", "flight": "Mã chuyến như VN122",
                 "seat": "Ghế như 12A", "booking_code": "Mã 4 ký tự như 4XJ2",
                 "origin": "Mã IATA 3 chữ cái", "destination": "Mã IATA 3 chữ cái", "method": "corp_card"}
        for p, rx in rules.items():
            v = args.get(p)
            if v is None:
                return {"status": "invalid_param", "param": p, "hint": f"Thiếu tham số {p}"}
            if allow_placeholder and isinstance(v, str) and v.startswith("$"):
                continue
            if not rx.fullmatch(str(v)):
                return {"status": "invalid_param", "param": p, "hint": hints[p]}
        return None


class LoopDetector:
    """Slide: so (tool, args) trong cửa sổ gần + đo đại lượng tiến triển. repeat_k=3 → báo ở lần gọi thứ ba."""

    def __init__(self, window=6, repeat_k=3, stall_n=4, exempt=("get_booking",)):
        self.recent = deque(maxlen=window)
        self.k, self.n, self.last, self.stall, self.exempt = repeat_k, stall_n, None, 0, set(exempt)

    def check(self, tool, args, progress):
        fp = (tool, repr(sorted(args.items())))
        if tool not in self.exempt and self.recent.count(fp) + 1 >= self.k:
            return "LOOP"
        self.recent.append(fp)
        self.stall = self.stall + 1 if progress == self.last else 0
        self.last = progress
        return "STALL" if self.stall >= self.n else None


@dataclass
class Budget:
    max_steps: int = 12
    max_model_calls: int = 12
    max_tokens: int = 40_000
    max_seconds: float = 60.0
    max_cost_usd: float = 0.05           # đơn giá giả định: 3$/1M token vào, 15$/1M token ra
    steps: int = 0
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    t0: float = field(default_factory=time.time)

    @property
    def cost(self):
        return self.tokens_in * 3e-6 + self.tokens_out * 15e-6

    def exceeded(self) -> Optional[str]:
        if self.steps >= self.max_steps:
            return "steps"
        if self.model_calls >= self.max_model_calls:
            return "model_calls"
        if self.tokens_in + self.tokens_out >= self.max_tokens:
            return "tokens"
        if time.time() - self.t0 >= self.max_seconds:
            return "seconds"
        if self.cost >= self.max_cost_usd:
            return "cost"
        return None


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)          # ước lượng thô: 4 ký tự ≈ 1 token


def msg_text(m: BaseMessage) -> str:
    t = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
    if isinstance(m, AIMessage) and m.tool_calls:
        t += json.dumps(m.tool_calls, ensure_ascii=False)
    return t


class Facts:
    """Sổ ghi sự thật rút ra từ observation (không phải từ lời model)."""

    def __init__(self, c: Constraints):
        self.c = c
        self.flights, self.seat_checks, self.eliminated = {}, {}, {}
        self.booking = None
        self.paid = self.confirmed = False
        self.stage, self.best_sat = 0, 0
        self.side_effects, self.tried, self.corpus = [], [], []

    def update(self, name, args, obs):
        self.corpus.append(json.dumps(obs, ensure_ascii=False))
        st = obs.get("status")
        if st != "ok":
            why = obs.get("error") or obs.get("reason") or obs.get("param") or st
            self.tried.append(f"{name}({', '.join(f'{k}={v}' for k, v in args.items())}) → {st}: {why}")
            if name == "check_seat" and st == "error":
                self.eliminated[args.get("flight")] = f"check_seat lỗi {why}"
            if name == "book_seat" and st in ("error", "denied"):
                self.eliminated[args.get("flight")] = f"book_seat {st}: {why}"
            return
        if name == "search_flights":
            for f in obs["flights"]:
                self.flights[f["flight"]] = f
                self.best_sat = max(self.best_sat, sum(self.c.checks(f).values()))
            if obs["flights"]:
                self.stage = max(self.stage, 1)
        elif name == "check_seat":
            fid = obs["flight"]
            self.seat_checks[fid] = obs
            f = self.flights.get(fid)
            if obs["seats"] == 0:
                self.eliminated[fid] = "hết chỗ"
            elif obs["price"] > self.c.max_price:
                self.eliminated[fid] = f"giá đổi {fmt(f['price']) if f else '?'} → {fmt(obs['price'])} vượt trần"
            else:
                self.stage = max(self.stage, 2)
        elif name == "book_seat":
            f = self.flights.get(args.get("flight"), {})
            self.booking = {"code": obs["booking_code"], "flight": args.get("flight"), "seat": args.get("seat"),
                            "price": obs["price"], "refundable": f.get("refundable", True)}
            self.stage = max(self.stage, 3)
            self.side_effects.append(f"book_seat {args.get('flight')} {args.get('seat')} → {obs['booking_code']} (held, {fmt(obs['price'])}đ)")
        elif name == "pay":
            self.paid = True
            self.stage = max(self.stage, 4)
            self.side_effects.append(f"pay {args.get('booking_code')} → paid")
        elif name == "get_booking":
            b = obs["booking"]
            if b["status"] == "confirmed" and b["paid"]:
                self.confirmed = True
                self.stage = max(self.stage, 5)

    def progress(self):
        return (self.stage, self.best_sat)

    def candidates(self, respect_time=True):
        return sorted((f for f in self.flights.values()
                       if f["flight"] not in self.eliminated and self.c.is_ok(f, ignore_time=not respect_time)),
                      key=lambda f: (f["price"], f["depart"]))


@dataclass
class ApprovalRequest:
    where: str
    action: str
    why: str

    def render(self):
        return f"Đang ở đâu: {self.where}\nĐịnh làm gì: {self.action}\nVì sao hỏi: {self.why}"


class Policy:
    """Kiểm quyền: chạy TRƯỚC khi thực thi hành động có tác dụng phụ (book_seat, pay)."""

    def __init__(self, c: Constraints):
        self.c = c
        self.approvals: set = set()

    def needs_approval(self, name, args, facts: Facts) -> Optional[str]:
        if name == "book_seat":
            fid = args.get("flight")
            f = facts.flights.get(fid, {})
            price = facts.seat_checks.get(fid, {}).get("price", f.get("price", 0))
            refundable = f.get("refundable", True)
        elif name == "pay" and facts.booking and args.get("booking_code") == facts.booking["code"]:
            fid, price, refundable = facts.booking["flight"], facts.booking["price"], facts.booking["refundable"]
        else:
            return None
        if fid in self.approvals:
            return None
        why = []
        if price > self.c.auto_limit:
            why.append(f"vượt hạn mức {fmt(self.c.auto_limit)}đ")
        if not refundable:
            why.append("vé không hoàn")
        return " và ".join(why) or None


@dataclass
class Handoff:
    termination: str
    reason: str
    done: list
    side_effects: list
    tried: list
    question: str

    def render(self) -> str:
        se = "; ".join(self.side_effects) or "chưa có hành động nào có tác dụng phụ"
        tried = "\n  - ".join(self.tried[-4:]) or "(chưa có hướng nào hỏng)"
        return (f"[BÀN GIAO · {self.termination}] {self.reason}\n"
                f"Đã làm: {'; '.join(self.done)}\nTác dụng phụ: {se}\nĐã thử:\n  - {tried}\nCâu hỏi: {self.question}")


NEG_OK = re.compile(r"(?i)\b(done|booked|đã đặt|đã xác nhận|đã thanh toán|confirmed|paid)\b")


def grounding_issues(text: str, facts: Facts) -> list:
    """Mọi mã chuyến, mã booking, số tiền, ghế trong câu trả lời phải có mặt trong observation đã nhận."""
    corpus = " ".join(facts.corpus)
    nums = {re.sub(r"\D", "", n) for n in re.findall(r"\d[\d.,]{4,}", corpus)}
    issues = []
    for code in sorted(set(re.findall(r"\b[A-Z]{2}\d{3,4}\b", text))):
        if code not in corpus:
            issues.append(f"mã chuyến {code} không có trong kết quả tool")
    for code in sorted(set(re.findall(r"\b\d[A-Z]{2}\d\b", text))):
        if code not in corpus:
            issues.append(f"mã đặt chỗ {code} không có trong kết quả tool")
    for seat in sorted(set(re.findall(r"\b\d{1,2}[A-F]\b", text))):
        if seat not in corpus:
            issues.append(f"ghế {seat} không có trong kết quả tool")
    for n in re.findall(r"\d[\d.,]{4,}", text):
        d = re.sub(r"\D", "", n)
        if len(d) >= 6 and d not in nums:
            issues.append(f"số tiền {n} không có trong kết quả tool")
    if NEG_OK.search(text) and not facts.confirmed:
        issues.append("tuyên bố đã đặt/thanh toán nhưng chưa có get_booking confirmed")
    return issues


@dataclass
class Outcome:
    obs: dict
    tag: str                      # ok | invalid | denied | error | need_approval
    stop: Optional[tuple] = None


@dataclass
class RunResult:
    termination: str
    reason: str
    answer: str
    answer_raw: str
    handoff: Optional[Handoff]
    trace: list
    facts: Facts
    approvals: set
    model_calls: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    replans: int
    nudges: int
    grounding_caught: list


# ════════════════════════════════════════════════════════════════════
# 4. MODEL GIẢ LẬP CÓ KIỂM SOÁT (để chạy offline, lặp lại được)
# ════════════════════════════════════════════════════════════════════
TOOLS_DOC = ("Tool: search_flights(origin,destination,date YYYY-MM-DD), check_seat(flight), "
             "book_seat(flight,seat), pay(booking_code,method), get_booking(booking_code).")


def system_prompt(mode: str, c: Constraints) -> str:
    base = (f"[MODE:{mode}]\nBạn là trợ lý đặt vé máy bay.\nRÀNG BUỘC (JSON): {c.model_dump_json()}\n"
            "Chỉ dùng thông tin lấy từ tool, không bịa mã chuyến, giá hay mã đặt chỗ.\n")
    if mode == "REACT":
        return base + "Mỗi lượt gọi tối đa một tool; khi đã xong thì trả lời ngắn gọn bằng tiếng Việt.\n" + TOOLS_DOC
    if mode == "PLAN":
        return base + (TOOLS_DOC + "\nHãy lập kế hoạch đầy đủ, trả về DUY NHẤT JSON "
                       '{"steps":[{"tool":"...","args":{...}}]}. Dùng placeholder "$best" (chuyến được chọn), '
                       '"$seat" (ghế trống đầu tiên), "$booking" (mã đặt chỗ) cho giá trị chưa biết.')
    if mode == "REPLAN":
        return base + (TOOLS_DOC + "\nKế hoạch cũ gặp vấn đề. Dựa vào FACTS, trả về DUY NHẤT JSON "
                       '{"steps":[...]} cho phần việc còn lại; trả {"steps":[]} nếu không còn cách nào.')
    return base + "Hãy báo kết quả cuối cho người dùng, ngắn gọn, chỉ dùng số liệu có trong kết quả tool."


@dataclass
class Obs:
    name: str
    args: dict
    res: dict


def extract_obs(msgs) -> list:
    calls, out = {}, []
    for m in msgs:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                calls[tc["id"]] = (tc["name"], tc["args"])
        elif isinstance(m, ToolMessage):
            name, args = calls.get(m.tool_call_id, (m.name, {}))
            try:
                res = json.loads(m.content)
            except Exception:
                res = {"status": "error", "error": "unparsable"}
            out.append(Obs(name, args, res))
    return out


def summarize(obs: list) -> str:
    getb = [o for o in obs if o.name == "get_booking" and o.res.get("status") == "ok"]
    if getb:
        b = getb[-1].res["booking"]
        if b["status"] == "confirmed" and b["paid"]:
            return (f"Đã đặt vé {b['flight']} khởi hành {b['depart']} ngày {b['date']}, ghế {b['seat']}, "
                    f"giá {fmt(b['price'])} VND, mã đặt chỗ {b['code']}; đã thanh toán.")
    bad = [o for o in obs if o.res.get("status") != "ok"]
    why = (bad[-1].res.get("error") or bad[-1].res.get("reason") or "không có chuyến thoả ràng buộc") if bad \
        else "không có chuyến thoả ràng buộc"
    return f"Chưa có vé nào được giữ chỗ hoặc thanh toán: {why}."


class SimModel(BaseChatModel):
    """Chính sách 'hợp lý' dựng bằng luật, cộng các lỗi model có chủ đích (flaws) để kiểm tra harness.
    Không phải model thật: kết quả đánh giá với SimModel đo harness và mẫu thiết kế, không đo chất lượng LLM."""
    flaws: set = Field(default_factory=set)

    @property
    def _llm_type(self) -> str:
        return "sim-flight-policy"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        sys_text = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        mode = re.search(r"\[MODE:(\w+)\]", sys_text).group(1)
        c = Constraints.model_validate_json(re.search(r"RÀNG BUỘC \(JSON\): (\{.*?\})\n", sys_text).group(1))
        ai = {"REACT": self._react, "PLAN": self._plan, "REPLAN": self._replan, "FINAL": self._final}[mode](c, messages)
        return ChatResult(generations=[ChatGeneration(message=ai)])

    # ---- tiện ích ----
    @staticmethod
    def _call(msgs, name, why, **args):
        return AIMessage(content=why, tool_calls=[{"name": name, "args": args, "id": f"call_{len(msgs)}", "type": "tool_call"}])

    def _react(self, c, msgs):
        fl, obs = self.flaws, extract_obs(msgs)
        call = lambda n, w, **a: self._call(msgs, n, w, **a)   # noqa: E731
        if not obs:
            return call("search_flights", "Cần tìm chuyến bay theo yêu cầu.", origin=c.origin,
                        destination=c.destination, date="07/10" if "bad_date" in fl else c.date)
        oks = [o for o in obs if o.name == "search_flights" and o.res.get("status") == "ok"]
        if not oks:
            last = obs[-1]
            m = re.search(r"\d{4}-\d{2}-\d{2}", last.res.get("hint", ""))
            date = m.group(0) if m else (last.args.get("date") if "loop_on_error" in fl else c.date)
            return call("search_flights", "Tìm lại với tham số đã sửa.", origin=c.origin, destination=c.destination, date=date)
        flights = {}
        for o in oks:
            for f in o.res["flights"]:
                flights[f["flight"]] = f
        if "hallucinate_final" in fl and not any(o.name == "book_seat" for o in obs) \
                and not any("VN999" in str(m.content) for m in msgs if isinstance(m, AIMessage)):
            return AIMessage(content="Done! Booked VN999, seat 5C, for 1,200,000 VND.")
        elim, errs = {}, Counter()
        for o in obs:
            fid = o.args.get("flight")
            if o.name == "check_seat":
                if o.res.get("status") == "ok":
                    if o.res["seats"] == 0:
                        elim[fid] = 1
                    elif o.res["price"] > c.max_price:
                        elim[fid] = 1
                elif o.res.get("status") == "error":
                    errs[fid] += 1
                    if "loop_on_error" not in fl and errs[fid] >= 2:
                        elim[fid] = 1
            if o.name == "book_seat" and o.res.get("status") != "ok":
                elim[fid] = 1
        booked = [o for o in obs if o.name == "book_seat" and o.res.get("status") == "ok"]
        if booked:
            code = booked[-1].res["booking_code"]
            if not any(o.name == "pay" and o.res.get("status") == "ok" for o in obs):
                return call("pay", "Thanh toán vé vừa giữ chỗ.", booking_code=code, method="corp_card")
            if not any(o.name == "get_booking" and o.res.get("status") == "ok" for o in obs):
                return call("get_booking", "Đọc lại để xác nhận.", booking_code=code)
            return AIMessage(content=summarize(obs))
        cands = sorted((f for f in flights.values() if c.is_ok(f, ignore_time="ignore_time" in fl) and f["flight"] not in elim),
                       key=lambda f: f["price"])
        if not cands:
            if "wander_on_empty" in fl:
                k = sum(1 for o in obs if o.name == "search_flights")
                offs = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]
                if k - 1 < len(offs):
                    d = (_date.fromisoformat(c.date) + timedelta(days=offs[k - 1])).isoformat()
                    return call("search_flights", "Thử ngày khác xem sao.", origin=c.origin, destination=c.destination, date=d)
            return AIMessage(content=summarize(obs))
        pick = cands[0]["flight"]
        chk = [o for o in obs if o.name == "check_seat" and o.args.get("flight") == pick
               and o.res.get("status") == "ok" and o.res["seats"] > 0]
        if not chk:
            return call("check_seat", f"Kiểm tra ghế chuyến rẻ nhất {pick}.", flight=pick)
        return call("book_seat", f"Giữ chỗ {pick}.", flight=pick, seat=chk[-1].res["available_seats"][0])

    def _std_steps(self, c, flight="$best", with_search_date=None, from_stage=0):
        s = []
        if with_search_date:
            s.append({"tool": "search_flights", "args": {"origin": c.origin, "destination": c.destination, "date": with_search_date}})
        s += [{"tool": "check_seat", "args": {"flight": flight}},
              {"tool": "book_seat", "args": {"flight": flight, "seat": "$seat"}},
              {"tool": "pay", "args": {"booking_code": "$booking", "method": "corp_card"}},
              {"tool": "get_booking", "args": {"booking_code": "$booking"}}]
        return s

    def _plan(self, c, msgs):
        fixed = any(isinstance(m, HumanMessage) and str(m.content).startswith("[HARNESS]") for m in msgs)
        date = "07/10" if ("bad_date" in self.flaws and not fixed) else c.date
        return AIMessage(content=json.dumps({"steps": self._std_steps(c, with_search_date=date)}, ensure_ascii=False))

    def _replan(self, c, msgs):
        facts = json.loads(str(msgs[-1].content).split("FACTS: ", 1)[1])
        le = facts.get("last_error")
        if "loop_on_error" in self.flaws and le:
            return AIMessage(content=json.dumps({"steps": [{"tool": le["tool"], "args": le["args"]}] + self._std_steps(c)[1:]}, ensure_ascii=False))
        if facts["stage"] >= 3 and facts.get("booking_code"):
            steps = ([] if facts["paid"] else [{"tool": "pay", "args": {"booking_code": facts["booking_code"], "method": "corp_card"}}])
            steps.append({"tool": "get_booking", "args": {"booking_code": facts["booking_code"]}})
        elif le and le["tool"] == "search_flights" and le["result"].get("status") != "ok":
            steps = self._std_steps(c, with_search_date=c.date)
        elif facts["candidates"]:
            steps = self._std_steps(c, flight=facts["candidates"][0])
        else:
            steps = []
        return AIMessage(content=json.dumps({"steps": steps}, ensure_ascii=False))

    def _final(self, c, msgs):
        if "hallucinate_final" in self.flaws:
            return AIMessage(content="Done! Booked VN999, seat 5C, for 1,200,000 VND.")
        return AIMessage(content=summarize(extract_obs(msgs)))


def parse_plan(text: str) -> Optional[list]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        steps = json.loads(m.group(0)).get("steps")
    except Exception:
        return None
    return steps if isinstance(steps, list) and all(isinstance(s, dict) for s in steps) else None


# ════════════════════════════════════════════════════════════════════
# 5. AGENT: BA MẪU THIẾT KẾ DỰNG BẰNG LANGGRAPH
# ════════════════════════════════════════════════════════════════════
class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    plan: list
    pc: int


class FlightAgent:
    def __init__(self, model, world: FlightWorld, pattern="react", constraints: Optional[Constraints] = None,
                 harness=True, approver: Optional[Callable] = None, budget: Optional[Budget] = None,
                 max_replans=3, max_nudges=2, verbose=False):
        assert pattern in ("react", "plan", "hybrid")
        self.world, self.pattern, self.harness, self.approver = world, pattern, harness, approver
        self.c = constraints or Constraints()
        self.max_replans, self.max_nudges, self.verbose = max_replans, max_nudges, verbose
        self.tools_list = make_tools(world)
        self.tools = {t.name: t for t in self.tools_list}
        self.plain_llm = model
        self.react_llm = model.bind_tools(self.tools_list)
        self.flaws = getattr(model, "flaws", set())
        # harness tắt: chỉ còn trần số lượt gọi model như ModelCallLimitMiddleware(run_limit=12)
        self.budget_cfg = budget or (Budget() if harness else Budget(max_steps=10**9, max_tokens=10**9,
                                                                    max_seconds=1e9, max_cost_usd=1e9))

    # ---------- trạng thái mỗi lần chạy ----------
    def _reset(self):
        self.facts = Facts(self.c)
        self.validator = Validator(self.tools)
        self.policy = Policy(self.c)
        self.loop = LoopDetector()
        self.budget = Budget(**{k: getattr(self.budget_cfg, k) for k in
                                ("max_steps", "max_model_calls", "max_tokens", "max_seconds", "max_cost_usd")})
        self.term = self.reason = None
        self.answer = self.answer_raw = ""
        self.trace, self.n_tool = [], 0
        self.nudges = self.replans = 0
        self.grounding_caught = []
        self.pending: Optional[ApprovalRequest] = None
        self.last: Optional[tuple] = None
        self.step_failed = None
        self.criteria_met = False
        self.chosen = None
        self.verified = None
        self.next_after_verify = None
        self.respect_time = self.harness or "ignore_time" not in self.flaws

    def log(self, line):
        self.trace.append(line)
        if self.verbose:
            print(line)

    def stop(self, kind, reason):
        if self.term is None:
            self.term, self.reason = kind, reason
            self.log(f"[HARNESS] DỪNG · {kind} · {reason}")

    # ---------- gọi model ----------
    def call_model(self, llm, msgs, label):
        b = self.budget
        b.model_calls += 1
        b.tokens_in += sum(est_tokens(msg_text(m)) for m in msgs)
        ai = llm.invoke(msgs)
        b.tokens_out += est_tokens(msg_text(ai))
        return ai

    def sys(self, mode):
        return system_prompt(mode, self.c)

    # ---------- thực thi tool có bảo vệ ----------
    def _execute(self, name, args) -> dict:
        t = self.tools.get(name)
        if not t:
            return {"status": "invalid_tool", "tool": name, "hint": f"Chỉ dùng: {sorted(self.tools)}"}
        try:
            return json.loads(t.invoke(args))
        except Exception as e:  # tham số sai kiểu/thiếu khi harness tắt
            return {"status": "error", "error": f"{type(e).__name__}"}

    def _constraint_guard(self, name, args) -> Optional[dict]:
        f = self.facts
        if name == "book_seat":
            fid = args["flight"]
            if fid not in f.flights:
                return {"status": "denied", "reason": f"{fid} không có trong kết quả tìm kiếm", "hint": "Chỉ đặt chuyến đã thấy từ search_flights"}
            chk = f.seat_checks.get(fid)
            if not chk or chk["seats"] == 0:
                return {"status": "denied", "reason": f"chưa check_seat thành công cho {fid}", "hint": "Gọi check_seat trước khi giữ chỗ"}
            v = self.c.violations(f.flights[fid], price=chk["price"])
            if v:
                return {"status": "denied", "reason": f"{fid} vi phạm ràng buộc: {', '.join(v)}", "hint": "Chọn chuyến khác thoả ràng buộc"}
        if name == "pay":
            if not f.booking or args["booking_code"] != f.booking["code"]:
                return {"status": "denied", "reason": "mã đặt chỗ không khớp booking đã tạo"}
            fl = f.flights.get(f.booking["flight"], {})
            if fl and self.c.violations(fl, price=f.booking["price"]):
                return {"status": "denied", "reason": "booking vi phạm ràng buộc, không thanh toán"}
        return None

    def guarded_execute(self, name, args) -> Outcome:
        self.budget.steps += 1
        args = dict(args)
        if self.harness:
            err = self.validator.check(name, args)
            if err:
                return self._record(name, args, err, "invalid")
            deny = self._constraint_guard(name, args)
            if deny:
                if name == "book_seat":
                    self.facts.eliminated[args["flight"]] = deny["reason"]
                return self._record(name, args, deny, "denied")
            why = self.policy.needs_approval(name, args, self.facts)
            if why:
                fl = args.get("flight") or (self.facts.booking or {}).get("flight")
                price = self.facts.seat_checks.get(fl, {}).get("price") or (self.facts.booking or {}).get("price", 0)
                req = ApprovalRequest(
                    where=f"đã có {len(self.facts.flights)} chuyến, chuyến chọn là {fl}",
                    action=f"{name}({', '.join(f'{k}={v}' for k, v in args.items())}) · {fmt(price)}đ",
                    why=why)
                self.pending = req
                if self.approver is None:
                    self.log(f"[V{self.n_tool + 1}] AI {name}({args}) → CHẶN, cần người duyệt: {why}")
                    return Outcome({"status": "pending_approval", "reason": why}, "need_approval", (Term.NEED_HUMAN, why))
                if self.approver(req):
                    self.policy.approvals.add(fl)
                else:
                    return self._record(name, args, {"status": "denied", "reason": "người dùng từ chối: " + why}, "denied")
        return self._record(name, args, self._execute(name, args), "ok")

    def _record(self, name, args, obs, tag) -> Outcome:
        self.n_tool += 1
        self.facts.update(name, args, obs)
        self.last = (name, args, obs, tag)
        short = json.dumps(obs, ensure_ascii=False)
        self.log(f"[V{self.n_tool}] TOOL {name}({', '.join(f'{k}={v}' for k, v in args.items())}) → {short[:150]}")
        return Outcome(obs, tag)

    # ---------- chọn chuyến, placeholder ----------
    def pick(self):
        if self.chosen:
            return self.facts.flights.get(self.chosen)
        cands = self.facts.candidates(respect_time=self.respect_time)
        if cands:
            self.chosen = cands[0]["flight"]
            return cands[0]
        return None

    def resolve(self, raw: dict):
        out, miss = {}, []
        for k, v in raw.items():
            if v == "$best":
                f = self.pick()
                out[k] = f["flight"] if f else None
                if not f:
                    miss.append("$best: không có chuyến thoả ràng buộc")
            elif v == "$booking":
                out[k] = self.facts.booking["code"] if self.facts.booking else None
                if not self.facts.booking:
                    miss.append("$booking: chưa có booking")
            elif v != "$seat":
                out[k] = v
        for k, v in raw.items():
            if v == "$seat":
                chk = self.facts.seat_checks.get(out.get("flight"), {})
                out[k] = (chk.get("available_seats") or ["12A"])[0]
        return out, miss

    def significant(self) -> Optional[str]:
        """Observation đổi đáng kể so với kế hoạch? (quyết định replan ở mẫu lai, quyết định hỏng ở mẫu plan thuần)"""
        name, args, obs, tag = self.last
        if obs.get("status") != "ok":
            return f"{name} → {obs.get('status')}: {obs.get('error') or obs.get('reason') or obs.get('param')}"
        if name == "search_flights" and not self.facts.candidates(self.respect_time):
            return "không có chuyến nào thoả đủ ràng buộc"
        if name == "check_seat":
            f = self.facts.flights.get(obs["flight"], {})
            if obs["seats"] == 0:
                return f"{obs['flight']} hết chỗ"
            if f and obs["price"] != f["price"]:
                return f"giá {obs['flight']} đổi {fmt(f['price'])} → {fmt(obs['price'])}"
        return None

    # ---------- bàn giao ----------
    def make_handoff(self) -> Handoff:
        f, c = self.facts, self.c
        done = [f"tìm được {len(f.flights)} chuyến", f"ràng buộc thoả tốt nhất {f.best_sat}/4",
                f"giai đoạn {f.stage}/5"]
        tried = list(f.tried) + [f"loại {k}: {v}" for k, v in f.eliminated.items()]
        q = {
            Term.NEED_HUMAN: f"Duyệt hành động này không? {self.pending.action} ({self.pending.why})" if self.pending else "Cần duyệt.",
            Term.LOOP: "Agent gọi lặp cùng một hành động với cùng kết quả. Đợi dịch vụ, thử chuyến khác hay hủy?",
            Term.STALL: f"Chưa chuyến nào thoả đủ ràng buộc ({f.best_sat}/4). Nới giờ bay, ngân sách (>{fmt(c.max_price)}đ) hay đổi ngày?",
            Term.BUDGET: f"Chạm trần {self.reason}. Cho thêm ngân sách để tiếp tục?",
        }[self.term]
        return Handoff(self.term, self.reason, done, list(f.side_effects), tried, q)

    def render_answer(self) -> str:
        b = self.verified
        return (f"Đã đặt vé {b['flight']} khởi hành {b['depart']} ngày {b['date']}, ghế {b['seat']}, "
                f"giá {fmt(b['price'])} VND, mã đặt chỗ {b['code']}; đã thanh toán.")

    def backend_done(self) -> bool:
        if not (self.facts.booking and self.facts.paid and self.facts.confirmed):
            self.missing_why = ["chưa đọc lại get_booking để xác nhận"] if self.facts.paid else ["chưa thanh toán"]
            return False
        b = self.world.get_booking(self.facts.booking["code"]).get("booking")   # harness tự đọc lại, không tin model
        ok, self.missing_why = completion_ok(b, self.c)
        if ok:
            self.verified = b
        return ok

    # ---------- các nút của đồ thị ----------
    def n_model(self, state):
        if not self.harness and self.budget.model_calls >= self.budget.max_model_calls:
            self.stop(Term.BUDGET, "model_calls (run_limit)")
            return {}
        ai = self.call_model(self.react_llm, [SystemMessage(self.sys("REACT"))] + state["messages"], "REACT")
        tc = ai.tool_calls
        self.log(f"[M{self.budget.model_calls}] AI " + (f"{tc[0]['name']}({tc[0]['args']})" if tc else f"(không gọi tool) {str(ai.content)[:100]}"))
        return {"messages": [ai]}

    def n_act(self, state):
        ai, out = state["messages"][-1], []
        for call in ai.tool_calls:
            o = self.guarded_execute(call["name"], call["args"])
            out.append(ToolMessage(content=json.dumps(o.obs, ensure_ascii=False), tool_call_id=call["id"], name=call["name"]))
            if o.stop:
                self.stop(*o.stop)
                break
        return {"messages": out}

    def n_plan(self, state):
        self.chosen = None
        msgs = [SystemMessage(self.sys("PLAN")), HumanMessage(GOAL)]
        steps, errs = None, []
        for _ in range(2 if self.harness else 1):
            ai = self.call_model(self.plain_llm, msgs, "PLAN")
            steps = parse_plan(str(ai.content))
            errs = self.validate_plan(steps) if self.harness else ([] if steps is not None else ["không parse được"])
            self.log(f"[M{self.budget.model_calls}] PLAN {len(steps or [])} bước" + (f" · bị từ chối: {errs}" if errs else ""))
            if not errs:
                break
            msgs += [ai, HumanMessage("[HARNESS] Kế hoạch bị từ chối: " + "; ".join(errs) + ". Hãy lập lại.")]
        if errs or steps is None:
            self.stop(Term.STALL, f"plan_invalid: {'; '.join(errs)}")
            return {}
        return {"plan": steps, "pc": 0}

    def validate_plan(self, steps) -> list:
        """Kế hoạch nhìn thấy được trước khi chạy → kiểm tĩnh và ước lượng chi phí (ưu thế của plan-then-execute)."""
        if not steps:
            return ["kế hoạch rỗng hoặc không parse được"]
        errs, seen = [], set()
        if len(steps) > self.budget.max_steps:
            errs.append(f"{len(steps)} bước vượt trần {self.budget.max_steps}")
        for i, s in enumerate(steps, 1):
            e = self.validator.check(s.get("tool"), s.get("args", {}), allow_placeholder=True)
            if e:
                errs.append(f"bước {i}: {e.get('param') or e['status']} — {e.get('hint')}")
            if s.get("tool") == "book_seat" and "check_seat" not in seen:
                errs.append(f"bước {i}: book_seat trước check_seat")
            if s.get("tool") == "pay" and "book_seat" not in seen:
                errs.append(f"bước {i}: pay trước book_seat")
            seen.add(s.get("tool"))
        return errs

    def n_execute(self, state):
        plan, pc = state["plan"], state["pc"]
        step = plan[pc]
        name = step.get("tool", "?")
        args, miss = self.resolve(step.get("args", {}))
        cid = f"plan_{self.n_tool + 1}"
        ai = AIMessage(content=f"[KẾ HOẠCH {pc + 1}/{len(plan)}] {name}",
                       tool_calls=[{"name": name, "args": {k: v for k, v in args.items() if v is not None}, "id": cid, "type": "tool_call"}])
        if miss:
            self.budget.steps += 1
            o = self._record(name, args, {"status": "error", "error": "unresolved_placeholder", "detail": miss}, "error")
        else:
            o = self.guarded_execute(name, args)
        if o.stop:
            self.stop(*o.stop)
        return {"messages": [ai, ToolMessage(content=json.dumps(o.obs, ensure_ascii=False), tool_call_id=cid, name=name)], "pc": pc + 1}

    def n_check(self, state):
        """Checklist sau mỗi observation, đúng thứ tự slide: 1 hoàn thành → 2 lặp → 3 bế tắc → 4 ngân sách."""
        if self.term or not self.harness and self.pattern == "react":
            return {}
        self.step_failed = self.significant() if self.pattern != "react" else None
        if not self.harness:
            return {}
        self.criteria_met = self.backend_done()
        if self.criteria_met:
            return {}
        name, args, obs, tag = self.last
        sig = self.loop.check(name, args, self.facts.progress())
        if sig == "LOOP":
            self.stop(Term.LOOP, f"{name}({args}) lặp lại {self.loop.k} lần")
        elif sig == "STALL":
            self.stop(Term.STALL, f"không tiến triển sau {self.loop.n} vòng, tiến độ {self.facts.progress()}")
        elif self.budget.exceeded():
            self.stop(Term.BUDGET, self.budget.exceeded())
        return {}

    def n_replan(self, state):
        if self.budget.model_calls >= self.budget.max_model_calls:
            self.stop(Term.BUDGET, "model_calls")
            return {}
        self.replans += 1
        self.chosen = None
        f = self.facts
        le = None
        if self.last and self.last[2].get("status") != "ok" or self.step_failed:
            le = {"tool": self.last[0], "args": self.last[1], "result": self.last[2]}
        facts = {"constraints": self.c.model_dump(), "stage": f.stage, "paid": f.paid,
                 "booking_code": f.booking["code"] if f.booking else None,
                 "candidates": [x["flight"] for x in f.candidates(self.respect_time)],
                 "eliminated": f.eliminated, "last_error": le, "reason": self.step_failed or self.missing_note()}
        msgs = [SystemMessage(self.sys("REPLAN")), HumanMessage("FACTS: " + json.dumps(facts, ensure_ascii=False))]
        steps, errs = None, []
        for _ in range(2 if self.harness else 1):
            ai = self.call_model(self.plain_llm, msgs, "REPLAN")
            steps = parse_plan(str(ai.content))
            errs = (self.validate_plan(steps) if steps else []) if self.harness else []
            if not errs:
                break
            msgs += [ai, HumanMessage("[HARNESS] Kế hoạch bị từ chối: " + "; ".join(errs))]
        self.log(f"[M{self.budget.model_calls}] REPLAN #{self.replans} ({facts['reason']}) → {len(steps or [])} bước")
        if errs:
            self.stop(Term.STALL, f"replan_invalid: {'; '.join(errs)}")
        elif not steps:
            self.stop(Term.STALL, "replan_infeasible: không còn hướng đi thoả ràng buộc")
        self.step_failed = None
        return {"plan": steps or [], "pc": 0}

    def missing_note(self):
        return "; ".join(getattr(self, "missing_why", [])) or "tiêu chí hoàn thành chưa đạt"

    def n_finalize(self, state):
        if self.budget.model_calls >= self.budget.max_model_calls and self.harness:
            self.stop(Term.BUDGET, "model_calls")
            return {}
        msgs = [SystemMessage(self.sys("FINAL"))] + state["messages"] + [HumanMessage("Hãy báo kết quả cho người dùng.")]
        ai = self.call_model(self.plain_llm, msgs, "FINAL")
        self.log(f"[M{self.budget.model_calls}] AI (trả lời cuối) {str(ai.content)[:100]}")
        return {"messages": [ai]}

    def n_verify(self, state):
        """Model/kế hoạch nói 'xong' → harness kiểm bằng code, không tin lời."""
        text = str(state["messages"][-1].content)
        self.answer_raw = text
        self.next_after_verify = None
        if not self.harness:
            self.answer = text
            self.stop(Term.DONE, "model dừng gọi tool (mặc định của framework)")
            return {}
        if self.backend_done():
            issues = grounding_issues(text, self.facts)
            if issues:
                self.grounding_caught = issues
                self.log(f"[HARNESS] grounding: {issues} → thay bằng câu trả lời dựng từ dữ liệu đã kiểm")
                text = self.render_answer()
            self.answer = text
            self.stop(Term.DONE, "tiêu chí hoàn thành đạt (đã kiểm bằng code)")
            return {}
        hope = bool(self.facts.candidates(self.respect_time)) or self.facts.stage >= 3
        self.log(f"[HARNESS] model báo xong nhưng chưa đạt: {self.missing_note()}")
        if self.pattern == "react" and hope and self.nudges < self.max_nudges and not self.budget.exceeded():
            self.nudges += 1
            self.next_after_verify = "model"
            return {"messages": [HumanMessage(f"[HARNESS] Chưa đạt tiêu chí hoàn thành ({self.missing_note()}). "
                                              "Tiếp tục dùng tool, đừng báo xong khi chưa có get_booking confirmed.")]}
        if self.pattern == "hybrid" and hope and self.replans < self.max_replans:
            self.next_after_verify = "replan"
            return {}
        self.stop(Term.STALL, "không còn hướng đi" if not hope else "đã báo xong nhưng tiêu chí chưa đạt")
        return {}

    # ---------- đồ thị ----------
    def build(self):
        g = StateGraph(AgentState)
        for n in ("check", "finalize", "verify"):
            g.add_node(n, getattr(self, "n_" + n))
        g.add_edge("finalize", "verify")
        if self.pattern == "react":
            g.add_node("model", self.n_model)
            g.add_node("act", self.n_act)
            g.add_edge(START, "model")
            g.add_conditional_edges("model", lambda s: END if self.term else ("act" if s["messages"][-1].tool_calls else "verify"),
                                    {"act": "act", "verify": "verify", END: END})
            g.add_edge("act", "check")
            g.add_conditional_edges("check", lambda s: END if self.term else ("finalize" if self.criteria_met else "model"),
                                    {"finalize": "finalize", "model": "model", END: END})
            g.add_conditional_edges("verify", lambda s: END if self.term else "model", {"model": "model", END: END})
        else:
            g.add_node("plan", self.n_plan)
            g.add_node("execute", self.n_execute)
            g.add_edge(START, "plan")
            g.add_conditional_edges("plan", lambda s: END if self.term else "execute", {"execute": "execute", END: END})
            g.add_edge("execute", "check")
            g.add_node("replan", self.n_replan)
            g.add_conditional_edges("replan", lambda s: END if self.term else "execute", {"execute": "execute", END: END})
            g.add_conditional_edges("check", self.route_plan_check,
                                    {"execute": "execute", "replan": "replan", "finalize": "finalize", END: END})
            g.add_conditional_edges("verify", lambda s: END if self.term else "replan", {"replan": "replan", END: END})
        return g.compile()

    def route_plan_check(self, s):
        if self.term:
            return END
        if self.step_failed and not (self.harness and self.criteria_met):
            if self.pattern == "plan":
                if self.harness:
                    self.stop(Term.STALL, f"plan_step_failed: {self.step_failed}")
                    return END
            else:
                if self.harness and self.replans >= self.max_replans:
                    self.stop(Term.STALL, f"replan_limit: {self.step_failed}")
                    return END
                return "replan"
        if self.criteria_met or s["pc"] >= len(s["plan"]):
            return "finalize"
        return "execute"

    def run(self, goal: str = GOAL) -> RunResult:
        self._reset()
        app = self.build()
        try:
            app.invoke({"messages": [HumanMessage(goal)]}, {"recursion_limit": 300})
        except Exception as e:                      # GraphRecursionError: trần của framework
            self.stop(Term.BUDGET, f"{type(e).__name__}")
        if self.term is None:
            self.stop(Term.STALL, "đồ thị kết thúc không có điều kiện dừng")
        handoff = self.make_handoff() if (self.harness and self.term != Term.DONE) else None
        answer = self.answer if self.term == Term.DONE else (handoff.render() if handoff else self.answer)
        b = self.budget
        return RunResult(self.term, self.reason, answer, self.answer_raw, handoff, self.trace, self.facts,
                         set(self.policy.approvals), b.model_calls, self.n_tool, b.tokens_in, b.tokens_out,
                         round(b.cost, 5), self.replans, self.nudges, self.grounding_caught)


# ════════════════════════════════════════════════════════════════════
# 6. CLI
# ════════════════════════════════════════════════════════════════════
def build_model(real: bool, flaws: set):
    if real:
        from langchain.chat_models import init_chat_model
        return init_chat_model(os.environ["SE373_MODEL"])
    return SimModel(flaws=set(flaws))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", choices=["react", "plan", "hybrid"], default="react")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="happy")
    ap.add_argument("--no-harness", action="store_true", help="tắt các lớp harness, chỉ còn trần số lượt gọi model")
    ap.add_argument("--approve", action="store_true", help="tự động duyệt mọi yêu cầu cần người duyệt")
    ap.add_argument("--real", action="store_true", help="dùng model thật từ biến môi trường SE373_MODEL")
    a = ap.parse_args()
    sc = SCENARIOS[a.scenario]
    world = FlightWorld(sc)
    agent = FlightAgent(build_model(a.real, sc.flaws), world, a.pattern, harness=not a.no_harness,
                        approver=(lambda req: True) if a.approve else None, verbose=True)
    print(f"# {a.pattern} · {sc.name} · {sc.desc} · harness={'tắt' if a.no_harness else 'bật'}")
    r = agent.run()
    print(f"\n=> {r.termination} ({r.reason})\n{r.answer}")
    print(f"model_calls={r.model_calls} tool_calls={r.tool_calls} tokens≈{r.tokens_in + r.tokens_out} cost≈${r.cost_usd}")


if __name__ == "__main__":
    main()
