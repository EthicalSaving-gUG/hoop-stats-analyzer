"""
SwissTransfer Link Resolver
===========================
Resolves SwissTransfer share links (https://www.swisstransfer.com/dl/<id>)
directly into high-speed presigned S3 video streaming URLs.
"""

import json
import re
import urllib.request
from typing import Optional, Tuple


def resolve_swisstransfer_url(url: str, timeout: int = 10) -> Tuple[Optional[str], Optional[dict]]:
    """
    Extracts the direct presigned download/streaming URL from a SwissTransfer link.

    Args:
        url: Full SwissTransfer share link (e.g. 'https://www.swisstransfer.com/dl/01a1dbe6-1d8b-73f7-a03c-05fe50a667e3')
        timeout: HTTP request timeout in seconds

    Returns:
        (direct_stream_url, metadata_dict) or (None, None) if resolution fails.
    """
    # 1. Extract link ID from URL
    match = re.search(r'/dl/([0-9a-fA-F-]+)', url)
    if not match:
        return None, None
    link_id = match.group(1)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }

    try:
        # 2. Fetch page HTML to get file and transfer metadata from Inertia props
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode('utf-8')

        script_match = re.search(r'<script data-page="app" type="application/json">(.*?)</script>', html, re.DOTALL)
        if not script_match:
            return None, None

        page_data = json.loads(script_match.group(1))
        props = page_data.get("props", {})
        transfer = props.get("transfer", {})
        files = transfer.get("files", [])
        if not files:
            return None, None

        primary_file = files[0]
        file_id = primary_file.get("id")
        file_name = primary_file.get("path") or primary_file.get("name")
        file_size = primary_file.get("size", 0)

        # 3. Call SwissTransfer API to get the presigned S3 URL
        api_url = f"https://www.swisstransfer.com/api/1/links/{link_id}/files/{file_id}"
        api_headers = {
            'User-Agent': headers['User-Agent'],
            'Accept': 'application/json'
        }
        api_req = urllib.request.Request(api_url, headers=api_headers)
        with urllib.request.urlopen(api_req, timeout=timeout) as api_resp:
            api_data = json.loads(api_resp.read().decode('utf-8'))

        stream_url = api_data.get("data", {}).get("url")
        metadata = {
            "link_id": link_id,
            "transfer_id": transfer.get("id"),
            "file_id": file_id,
            "filename": file_name,
            "size_bytes": file_size,
            "size_gb": round(file_size / (1024 ** 3), 2),
            "mime_type": primary_file.get("mime_type"),
            "expires_at": transfer.get("expires_at")
        }
        return stream_url, metadata

    except Exception as exc:
        print(f"[SwissTransferResolver] Error resolving URL {url}: {exc}")
        return None, None
