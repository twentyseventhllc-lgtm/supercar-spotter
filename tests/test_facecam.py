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
