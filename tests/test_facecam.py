# tests/test_facecam.py
def test_role_and_color_admin():
    import facecam
    role, color = facecam.role_and_color(True)
    assert role == "admin" and color == facecam.YELLOW

def test_role_and_color_unknown():
    import facecam
    role, color = facecam.role_and_color(False)
    assert role == "unknown" and color == facecam.BLUE

def test_tag_is_ww():
    import facecam
    assert facecam.TAG == "WW"

def test_image_files_filters_and_sorts(tmp_path):
    import os
    import facecam
    for name in ["a.jpg", "b.PNG", "c.jpeg", "notes.txt", ".DS_Store"]:
        (tmp_path / name).write_bytes(b"x")
    got = [os.path.basename(p) for p in facecam._image_files(str(tmp_path))]
    assert got == ["a.jpg", "b.PNG", "c.jpeg"]   # images only, sorted; non-images dropped
