"""
유틸리티 함수들
파일 읽기, 코드 블록 감지, 토큰 추출 등의 범용 함수들
"""
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any
from pathlib import Path


def read_text(path: Path) -> str:
    """파일을 UTF-8로 읽어서 텍스트 반환"""
    return path.read_text(encoding="utf-8", errors="ignore")


def detect_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    텍스트에서 코드 블록과 일반 텍스트를 분리
    
    Returns:
        List[Tuple[str, str]]: ("code" | "text", content) 형태의 리스트
    """
    blocks: List[Tuple[str, str]] = []
    fence_pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", re.DOTALL)
    pos = 0

    # 먼저 fenced 코드 블록을 전부 찾아서 처리
    matches = list(fence_pattern.finditer(text))
    if matches:
        for m in matches:
            before = text[pos:m.start()].strip()
            if before:
                blocks.append(("text", before))
            code = m.group(1)
            blocks.append(("code", code))
            pos = m.end()
        tail = text[pos:].strip()
        if tail:
            blocks.append(("text", tail))
        return blocks

    # fallback: 간이 코드 감지
    lines = text.splitlines()
    buf: List[str] = []
    mode = "text"

    def flush():
        nonlocal buf, mode
        if buf:
            blocks.append((mode, "\n".join(buf).strip()))
            buf = []

    codey = re.compile(r"^(\s*(def |class |import |from |if __name__ == '__main__':|for |while |try:|except |with ))")
    for ln in lines:
        if codey.search(ln):
            if mode != "code":
                flush()
                mode = "code"
            buf.append(ln)
        else:
            if mode != "text":
                flush()
                mode = "text"
            buf.append(ln)
    flush()
    return blocks


def extract_allowed_tokens(text: str) -> Dict[str, Any]:
    """
    텍스트에서 허용된 모듈과 심볼들을 추출
    
    Returns:
        Dict[str, Any]: {"modules": List[str], "symbols": List[str]}
    """
    imports = re.findall(r"^(?:from\s+([\w\.]+)\s+import\s+([\w\*,\s]+)|import\s+([\w\.,\s]+))", text, re.M)
    modules = set()
    names = set()
    
    for a, b, c in imports:
        if a:
            modules.add(a)
            for nm in re.split(r"\s*,\s*", b.strip()):
                if nm:
                    names.add(nm)
        elif c:
            for m in re.split(r"\s*,\s*", c.strip()):
                if m:
                    modules.add(m)

    funcs = re.findall(r"^\s*def\s+([a-zA-Z_][\w]*)\(", text, re.M)
    klass = re.findall(r"^\s*class\s+([A-Z][\w]*)\(", text, re.M)
    consts = re.findall(r"^([A-Z_]{3,})\s*=\s*", text, re.M)

    return {
        "modules": sorted(modules),
        "symbols": sorted(set(funcs) | set(klass) | set(consts)),
    }


def find_unknown_tokens(code: str, allowed: Dict[str, Any]) -> List[str]:
    """
    코드에서 허용되지 않은 토큰들을 찾아 반환
    
    Args:
        code: 검사할 코드 문자열
        allowed: extract_allowed_tokens()의 결과
        
    Returns:
        List[str]: 미허용 토큰들 ("module:name" 또는 "symbol:name" 형태)
    """
    unknown = []
    
    for m in re.finditer(r"^(?:from\s+([\w\.]+)\s+import\s+([\w\*,\s]+)|import\s+([\w\.,\s]+))", code, re.M):
        mod_from, names, mods = m.groups()
        
        if mod_from and mod_from not in allowed["modules"]:
            unknown.append(f"module:{mod_from}")
            
        if names:
            for nm in re.split(r"\s*,\s*", names.strip()):
                nm = nm.strip()
                if nm and nm not in allowed["symbols"]:
                    unknown.append(f"symbol:{nm}")
                    
        if mods:
            for md in re.split(r"\s*,\s*", mods.strip()):
                md = md.strip()
                if md and md not in allowed["modules"]:
                    unknown.append(f"module:{md}")
                    
    return sorted(set(unknown))