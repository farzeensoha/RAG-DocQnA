from fastapi import Request, HTTPException


def current_user(request: Request):
    """Returns the session's user dict ({id, username, role}) or None."""
    return request.session.get("user")


def require_login(request: Request):
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in.")
    return user


def require_admin(request: Request):
    user = require_login(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user
