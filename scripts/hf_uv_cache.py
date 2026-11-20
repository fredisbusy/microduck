#!/usr/bin/env python3
"""uv 라이브러리 캐시를 Hugging Face 버킷에 보관해서 Colab·Kaggle·HF Jobs 세션 사이에 재사용한다.

노트북 1-2 셀이 GitHub에서 받아 `uv sync` 앞뒤로 부른다. 버킷은 내 계정의 비공개 `microduck-uv-cache`.

    python hf_uv_cache.py restore UV_LOCK CACHE_DIR   # 버킷의 캐시를 받아 CACHE_DIR에 풀기
    python hf_uv_cache.py save    UV_LOCK CACHE_DIR   # 이번 키의 캐시가 버킷에 없으면 tar로 올리고 예전 것 지우기

키 = CPU 종류 + uv 버전(UV_VERSION) + uv.lock 해시. 공식 코드의 uv.lock이 바뀌면 restore는 가장 최근
캐시로 시작하고(대부분 그대로 쓰임), save가 새 키로 올린 뒤 예전 것을 지운다.
캐시가 실패해도 설치는 평소처럼 진행되도록 항상 0으로 끝난다. HF_TOKEN(Write)이 필요하다.
"""
import hashlib
import os
import platform
import subprocess
import sys
import time

BUCKET_NAME = os.environ.get("UV_CACHE_BUCKET", "microduck-uv-cache")
PREFIX = f"uv-cache-{platform.machine()}-"


def cache_name(lock_path):
    h = hashlib.sha256(open(lock_path, "rb").read())
    h.update(os.environ.get("UV_VERSION", "").encode())
    return f"{PREFIX}{h.hexdigest()[:16]}.tar"


def newest(files):
    return max(files, key=lambda f: (f.uploaded_at or f.mtime).timestamp() if (f.uploaded_at or f.mtime) else 0)


def run(mode, lock_path, cache_dir, api):
    bucket = f"{api.whoami()['name']}/{BUCKET_NAME}"
    api.create_bucket(bucket, private=True, exist_ok=True)
    name = cache_name(lock_path)
    files = [f for f in api.list_bucket_tree(bucket, recursive=True)
             if getattr(f, "type", "") == "file" and f.path.startswith(PREFIX)]
    exact = next((f for f in files if f.path == name), None)
    cache_dir = os.path.abspath(cache_dir)
    tmp = cache_dir.rstrip("/") + ".tar"   # 캐시 폴더 옆 (같은 디스크)
    t0 = time.time()

    if mode == "restore":
        pick = exact or (newest(files) if files else None)
        if pick is None:
            print("📦 HF 캐시 없음: 이번엔 전부 내려받고, 끝나면 다음 세션용으로 HF 버킷에 저장해요")
            return
        note = "" if exact else " (공식 코드가 바뀌어서 지난 캐시부터 시작)"
        print(f"📦 HF 캐시 받는 중: {pick.size / 1e9:.1f} GB{note}")
        api.download_bucket_files(bucket, [(pick, tmp)], raise_on_missing_files=True)
        os.makedirs(cache_dir, exist_ok=True)
        subprocess.run(["tar", "-xf", tmp, "-C", cache_dir], check=True)
        os.remove(tmp)
        print(f"   받아서 풀기 {time.time() - t0:.0f}초")
    elif mode == "save":
        if exact:
            print("📦 HF 캐시가 이미 최신이에요 (올릴 것 없음)")
            return
        env = dict(os.environ, UV_CACHE_DIR=cache_dir)
        subprocess.run(["uv", "cache", "prune"], env=env, check=False, capture_output=True)
        subprocess.run(["tar", "-cf", tmp, "-C", cache_dir, "."], check=True)
        size = os.path.getsize(tmp)
        print(f"☁️ 다음 세션용 캐시를 HF 버킷 {bucket}에 올리는 중: {size / 1e9:.1f} GB (처음 한 번)")
        try:
            api.batch_bucket_files(bucket, add=[(tmp, name)], delete=[f.path for f in files])
        finally:
            os.remove(tmp)
        print(f"   올리기 {time.time() - t0:.0f}초")
    else:
        raise SystemExit(f"모르는 명령: {mode}")


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        return 0
    mode, lock_path, cache_dir = argv[1:]
    try:
        from huggingface_hub import HfApi
        run(mode, lock_path, cache_dir, HfApi())
    except Exception as e:  # 캐시는 있으면 좋은 것: 실패해도 설치는 계속
        print(f"⚠️ HF 캐시 {mode} 건너뜀, 설치는 평소대로 계속 ({type(e).__name__}: {str(e)[:120]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
