# notify.py — push a phone notification (with the photo) when a supercar is caught.
#
# Uses ntfy.sh: free, no account. To receive these:
#   1. Install the "ntfy" app (iOS App Store / Google Play).
#   2. In the app, Subscribe to the topic in NTFY_TOPIC below.
#   3. Run the spotter — caught cars push straight to your phone.
#
# Only the standard library is used (urllib), so there's nothing extra to install.
import os
import time
import urllib.request

# ─── config ───────────────────────────────────────────────────
ENABLED = True
NTFY_TOPIC = "supercar-spotter-207d48eb"   # your private channel — subscribe to this
NTFY_SERVER = "https://ntfy.sh"
NOTIFY_COOLDOWN = 60   # min seconds between phone pushes. Without this we fire one
                       # push per catch; a busy street (or false positives) floods
                       # ntfy/Apple's push service, which then silently drops them.
# ──────────────────────────────────────────────────────────────

# Track ids already pushed, so one car = one notification (not one per frame).
_notified = set()
_last_push_ts = None   # wall-clock of the last actual push (None = never pushed yet)


def _ascii(value):
    # HTTP headers must be ASCII/latin-1. Drop anything else so a stray unicode
    # char never crashes the send. The emoji comes from the `Tags` shortcode.
    return str(value).encode("ascii", "ignore").decode("ascii")


def _build_request(topic, title, message, image_bytes, filename,
                   tags="racing_car", priority="high", server=NTFY_SERVER):
    # The image is the request body (ntfy shows it inline); text rides in headers.
    return urllib.request.Request(
        f"{server}/{topic}",
        data=image_bytes,
        method="PUT",
        headers={
            "Filename": _ascii(filename),
            "Title": _ascii(title),
            "Message": _ascii(message),
            "Tags": _ascii(tags),
            "Priority": _ascii(priority),
        },
    )


def ntfy_photo(topic, title, message, image_path, timeout=10):
    with open(image_path, "rb") as fh:
        image_bytes = fh.read()
    req = _build_request(topic, title, message, image_bytes,
                         os.path.basename(image_path))
    urllib.request.urlopen(req, timeout=timeout)


def notify_catch(action, image_path, now=None):
    """Push one notification the first time a track is caught, rate-limited to one
    push per NOTIFY_COOLDOWN seconds so a burst of catches can't flood (and get
    silently throttled by) the delivery chain. Network failures are swallowed (a
    missed ping must never crash the live loop)."""
    global _last_push_ts
    if not ENABLED or not NTFY_TOPIC or not image_path:
        return False
    if action.track_id in _notified:
        return False
    t = now if now is not None else time.time()
    if _last_push_ts is not None and t - _last_push_ts < NOTIFY_COOLDOWN:
        return False              # cooldown: drop WITHOUT marking — may push later
    _notified.add(action.track_id)
    _last_push_ts = t
    title = f"{action.label} {action.confidence:.0%}"
    try:
        ntfy_photo(NTFY_TOPIC, title, "Supercar Spotter caught one!", image_path)
        print(f"📲 pushed {title} to your phone")
        return True
    except Exception as exc:                      # noqa: BLE001 - never crash the loop
        print(f"(notify failed: {exc})")
        return False
