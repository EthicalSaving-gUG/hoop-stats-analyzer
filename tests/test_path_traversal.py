import pytest
from fastapi.testclient import TestClient
import os

from server import app, UPLOAD_DIR

client = TestClient(app)

def test_secure_filename_path_traversal(tmpdir):
    # Ensure the upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Simulate an upload with a path traversal filename
    malicious_filename = "../../evil.sh"

    # Create a dummy file to upload
    file_content = b"echo 'hack'"

    response = client.post(
        "/manual/upload-video",
        files={"file": (malicious_filename, file_content, "application/octet-stream")}
    )

    assert response.status_code == 200
    data = response.json()

    # The secure_filename should extract "evil.sh". It allows . in filenames.
    expected_safe_filename = "evil.sh"

    assert expected_safe_filename in data["video_url"]

    # Verify the file was created in the correct directory with the safe name
    expected_file_path = os.path.join(UPLOAD_DIR, expected_safe_filename)
    assert os.path.exists(expected_file_path)

    # Read back to ensure it's the right file
    with open(expected_file_path, "rb") as f:
        assert f.read() == file_content

    # Cleanup
    if os.path.exists(expected_file_path):
        os.remove(expected_file_path)

def test_secure_filename_normal():
    # Ensure the upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    normal_filename = "test_video.mp4"
    file_content = b"fake video content"

    response = client.post(
        "/manual/upload-video",
        files={"file": (normal_filename, file_content, "video/mp4")}
    )

    assert response.status_code == 200
    data = response.json()

    # The secure_filename should extract "test_video.mp4". It allows . in filenames.
    expected_safe_filename = "test_video.mp4"

    assert expected_safe_filename in data["video_url"]

    expected_file_path = os.path.join(UPLOAD_DIR, expected_safe_filename)
    assert os.path.exists(expected_file_path)

    # Cleanup
    if os.path.exists(expected_file_path):
        os.remove(expected_file_path)
