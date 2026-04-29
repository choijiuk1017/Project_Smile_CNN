import json
import random

scene_types = {
    "spawn_corpse": {
        "rag": "방 안에 시체가 쓰러져 있다.",
        "reference": "저 시체는 뭐지? 무언가한테 공격받은 것 같은데...",
        "answers": [
            "피가 주변에 튄 걸 보면... 저 사람은 공격받은 건가?",
            "움직임이 없다... 이미 늦은 건가?",
            "저 상태라면 단순히 쓰러진 건 아닌 것 같다...",
            "피 흔적이 이상하다... 누군가에게 당한 건가?",
            "여기서 무슨 일이 벌어진 거지...?"
        ]
    },
    "hallway_corpses": {
        "rag": "복도에 여러 시체와 핏자국이 있다.",
        "reference": "끔찍하군... 대체 무슨 일이 있었던 거야?",
        "answers": [
            "복도에 쓰러진 사람이 한둘이 아니다... 여기서 무슨 일이 있었던 거지?",
            "이 정도면 우연이라고 보기 어렵다... 집단으로 당한 건가?",
            "여기 전체가 사건 현장 같은데... 누가 이런 짓을?",
            "사람들이 이렇게까지 쓰러져 있다니... 뭔가 터진 건가?",
            "이건 단순 사고가 아니라... 뭔가 계획된 일 같은데?"
        ]
    },
    "locked_private_door": {
        "rag": "PRIVATE 표시가 있는 문과 인증 장치가 있다.",
        "reference": "문이 잠긴 것 같은데... 옆에 키패드를 보면 키카드 같은 것이 필요하겠어...",
        "answers": [
            "PRIVATE 표시에 장치까지 붙어 있다... 그냥 열리는 문은 아닌 것 같다.",
            "옆에 있는 장치를 보면... 인증 없이는 못 여는 구조인가?",
            "문 상태를 보니 잠겨 있는 것 같다... 열 방법이 따로 있나?",
            "이 장치는 출입을 제한하는 것 같은데... 키카드가 필요한 건가?",
            "이건 평범한 문이 아니다... 뭔가 조건이 필요한 것 같다."
        ]
    },
    "rest_area": {
        "rag": "침대와 가구가 있는 방은 직원 휴게 공간처럼 보인다.",
        "reference": "이곳은 직원들 휴게 공간인가 보군... 여기 어딘가에 필요한 물건이 있을지도 몰라.",
        "answers": [
            "침대와 가구를 보면... 사람들이 머물던 공간 같은데?",
            "여기 상태를 보니 급하게 떠난 것 같다... 뭔가 남아 있을까?",
            "이 정도면 단순 방은 아니다... 필요한 물건이 있을 수도 있다.",
            "이 공간은 누군가 사용하던 흔적이 있다... 확인해봐야 하나?",
            "여긴 그냥 지나칠 곳은 아닌 것 같다... 뭔가 숨겨져 있을지도."
        ]
    },
    "office_desk": {
        "rag": "책상과 컴퓨터가 있는 사무 공간이다.",
        "reference": "책상 주변에 단서가 남아 있을지도 모른다.",
        "answers": [
            "책상 주변에 물건들이 흩어져 있다... 뭔가 남아 있을지도 모른다.",
            "컴퓨터와 책상이 있는 걸 보면... 기록이 남아 있을까?",
            "이건 작업 공간 같은데... 단서가 숨겨져 있을 수도 있겠어.",
            "누군가 여기서 일하다 떠난 것 같다... 확인해볼 필요가 있다.",
            "흩어진 물건들을 보면... 그냥 넘길 상황은 아닌 것 같다."
        ]
    },
    "blood_stains_only": {
        "rag": "핏자국만 있고 시체는 보이지 않는다.",
        "reference": "핏자국은 보이지만, 이것만으로는 확실히 알 수 없다.",
        "answers": [
            "핏자국은 남아 있는데... 당사자는 보이지 않는다.",
            "여기서 무언가 있었던 건 맞지만... 확실한 건 없다.",
            "흔적만 남아 있다... 이걸로 판단하긴 어렵다.",
            "피는 있는데 사람이 없다... 어디로 간 거지?",
            "이건 단서라고 보기엔 부족하다... 더 봐야 할 것 같다."
        ]
    },
    "no_clue": {
        "rag": "단서 없음",
        "reference": "지금 보이는 것만으로는 확실한 단서를 찾기 어렵다.",
        "answers": [
            "지금 상태로는 뭔가 판단하기 어렵다.",
            "확실한 단서는 보이지 않는다.",
            "이 장면만으로는 알 수 있는 게 없다.",
            "아직 판단할 정보가 부족하다.",
            "여기서는 얻을 수 있는 게 없어 보인다."
        ]
    }
}

scenes = [
    "A person is lying on the floor in a dark room.",
    "A dim corridor with bodies lying on the ground.",
    "A PRIVATE door with a keypad device.",
    "A room with beds and furniture.",
    "A desk with scattered items.",
    "A room with blood stains but no body."
]

data = []

for _ in range(300):
    st = random.choice(list(scene_types.keys()))
    info = scene_types[st]

    data.append({
        "scene": random.choice(scenes),
        "rag": info["rag"],
        "reference_answer": info["reference"],
        "answer": random.choice(info["answers"])
    })

with open("lora_300_reasoning.jsonl", "w", encoding="utf-8") as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")