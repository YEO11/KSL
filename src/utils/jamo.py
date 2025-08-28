# Korean fingerspelling classes (consonants + vowels)
JAMO_CLASSES = [
    # Consonants (14)
    "giyok","nieun","digeut", # "ㄹ","ㅁ","ㅂ","ㅅ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ",
    # Vowels 
    # "ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ",

    # 쌍자음 (21)
    # "ㄲ","ㄸ","ㅃ","ㅆ","ㅉ"
]

ID2NAME = {i: n for i, n in enumerate(JAMO_CLASSES)}
NAME2ID = {n: i for i, n in ID2NAME.items()}

def save_names_txt(path: str):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name in JAMO_CLASSES:
            f.write(f"{name}\n")
