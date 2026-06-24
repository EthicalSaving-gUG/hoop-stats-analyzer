import os
import pytest
from fastapi.testclient import TestClient
from server import app, UPLOAD_DIR

client = TestClient(app)

def test_file_upload_path_traversal():
    """Test that a file uploaded with a path traversal filename is sanitized."""
    traversal_filename = "../../../evil_file.txt"
    file_content = b"evil content"

    # Send a file to the /analyze/file endpoint
    response = client.post(
        "/analyze/file",
        files={"file": (traversal_filename, file_content, "text/plain")}
    )

    assert response.status_code == 200

    # Determine what the sanitized filename should be
    expected_filename = "evil_file.txt"
    expected_file_path = os.path.join(UPLOAD_DIR, expected_filename)

    # Check that the file was created in the correct location with the sanitized name
    assert os.path.exists(expected_file_path)

    # Check that the file content matches
    with open(expected_file_path, "rb") as f:
        assert f.read() == file_content

    # Check that the original path traversal payload did not write outside the upload dir
    # (Checking a directory up just in case, though it's hard to assert globally)
    traversal_path = os.path.join(UPLOAD_DIR, traversal_filename)
    # The literal traversal path should not exist as a file if os.path.join resolved it outside
    # Actually, we can check if it exists in the parent directory:
    parent_dir_evil_file = os.path.join(os.path.dirname(os.path.abspath(UPLOAD_DIR)), "evil_file.txt")
    if os.path.abspath(UPLOAD_DIR) != os.path.dirname(os.path.abspath(UPLOAD_DIR)):
        assert not os.path.exists(parent_dir_evil_file)

    # Clean up
    if os.path.exists(expected_file_path):
        os.remove(expected_file_path)

def test_secure_filename_empty():
    """Test that empty filename is handled."""
    response = client.post(
        "/manual/upload-video",
        files={"file": (" ", b"empty content", "video/mp4")}
    )
    assert response.status_code == 200
    expected_file_path = os.path.join(UPLOAD_DIR, "uploaded_file")
    assert os.path.exists(expected_file_path)
    if os.path.exists(expected_file_path):
        os.remove(expected_file_path)

def test_secure_filename_complex_characters():
    """Test handling of weird characters."""
    complex_name = "C:\\Windows\\System32\\cmd.exe"
    response = client.post(
        "/manual/upload-video",
        files={"file": (complex_name, b"some content", "video/mp4")}
    )
    assert response.status_code == 200

    # Sanitized result for "C:\Windows\System32\cmd.exe"
    expected_file_path = os.path.join(UPLOAD_DIR, "cmd.exe")
    assert os.path.exists(expected_file_path)
    if os.path.exists(expected_file_path):
        os.remove(expected_file_path)
