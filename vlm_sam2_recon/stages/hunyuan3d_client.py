"""Hunyuan-3D (腾讯云混元生3D) external-API client for mesh reconstruction.

Wired for Tencent Cloud's OpenAI-compatible 3D gateway
(https://tokenhub.tencentmaas.com), which authenticates with a single Bearer
API key — the flow is submit -> poll -> download:

  POST /v1/api/3d/submit  {model, image_base64, enable_pbr, face_count, ...}
      -> {"id": ..., "status": "queued"}
  POST /v1/api/3d/query   {model, id}
      -> {"status": "completed", "data": [{"type": "glb"|"obj", "url": ...}]}

Config is read from CLI flags or env vars (see `Hunyuan3DClientConfig.from_env`):
  HUNYUAN3D_BASE_URL   default https://tokenhub.tencentmaas.com
  HUNYUAN3D_API_KEY    your secret key (never hard-code it)
  HUNYUAN3D_MODEL      default hy-3d-3.0 (or hy-3d-3.1)

If your API access is the *native* Tencent action (ai3d.tencentcloudapi.com,
SecretId + SecretKey + TC3-HMAC-SHA256 signing) rather than this Bearer-key
gateway, `_auth_headers` / `submit_job` / `poll_job` are the three spots to
swap. Nothing here imports the rest of the pipeline, so it tests standalone.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class Hunyuan3DError(RuntimeError):
    """Raised for any Hunyuan3D API failure (config, HTTP, or job error)."""


class Hunyuan3DSubmitError(Hunyuan3DError):
    """Raised when submit returns a terminal failed response."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        message = error.get("message") or data
        code = error.get("code")
        suffix = f" code={code}" if code else ""
        super().__init__(f"submit_job failed:{suffix} {message}")


@dataclass
class Hunyuan3DClientConfig:
    # Tencent Cloud Hunyuan-3D OpenAI-compatible gateway defaults.
    base_url: str | None = "https://tokenhub.tencentmaas.com"
    api_key: str | None = None
    model: str | None = "hy-3d-3.0"  # or "hy-3d-3.1"
    # Hunyuan submit knobs (see SubmitHunyuanTo3DProJob docs).
    enable_pbr: bool = True          # PBR materials -> finer textured mesh
    face_count: int | None = 500000  # [3000, 1500000]; higher = denser mesh
    generate_type: str | None = None  # Normal / LowPoly / Geometry / Sketch
    result_format: str | None = None  # OBJ/GLB come back by default; STL/USDZ/FBX optional
    # Submit/poll/download tuning — safe defaults, override via CLI/env if needed.
    timeout_sec: float = 60.0
    poll_interval_sec: float = 5.0
    poll_timeout_sec: float = 1200.0
    max_retries: int = 3
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Free-form knobs merged verbatim into the submit payload.
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> "Hunyuan3DClientConfig":
        cfg = cls()
        if os.environ.get("HUNYUAN3D_BASE_URL"):
            cfg.base_url = os.environ["HUNYUAN3D_BASE_URL"]
        if os.environ.get("HUNYUAN3D_API_KEY"):
            cfg.api_key = os.environ["HUNYUAN3D_API_KEY"]
        if os.environ.get("HUNYUAN3D_MODEL"):
            cfg.model = os.environ["HUNYUAN3D_MODEL"]
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def require_configured(self) -> None:
        env_names = {"base_url": "HUNYUAN3D_BASE_URL", "api_key": "HUNYUAN3D_API_KEY"}
        missing = [env_names[name] for name in ("base_url", "api_key") if not getattr(self, name)]
        if missing:
            raise Hunyuan3DError(
                "Hunyuan3D API not configured; set " + " and ".join(missing)
                + " (or pass --base-url / --api-key). "
                "Fill in your endpoint details in hunyuan3d_client.py (# TODO(api))."
            )


@dataclass
class Hunyuan3DResult:
    """Outcome of one image -> mesh reconstruction call."""

    mesh_path: Path
    task_id: str | None = None
    mesh_format: str = "glb"
    raw_response: dict[str, Any] = field(default_factory=dict)


def _lazy_requests():
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        raise Hunyuan3DError("The 'requests' package is required for the Hunyuan3D client.") from exc
    return requests


class Hunyuan3DClient:
    """Thin submit -> poll -> download client around a Hunyuan3D REST API.

    The generic flow below matches most async 3D-generation APIs. If your API
    is synchronous (returns the mesh URL directly from submit), just make
    `poll_job` return that response unchanged. Everything specific to *your*
    endpoint lives in the three `# TODO(api)` blocks.
    """

    def __init__(self, config: Hunyuan3DClientConfig) -> None:
        self.config = config
        self._session = None

    # -- infrastructure (safe to leave as-is) -----------------------------

    @property
    def session(self):
        if self._session is None:
            requests = _lazy_requests()
            self._session = requests.Session()
        return self._session

    def _url(self, path: str) -> str:
        base = (self.config.base_url or "").rstrip("/")
        return path if path.startswith("http") else f"{base}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs: Any):
        requests = _lazy_requests()
        url = self._url(path)
        headers = {**self._auth_headers(), **self.config.extra_headers, **kwargs.pop("headers", {})}
        kwargs.setdefault("timeout", self.config.timeout_sec)
        last_error = "unknown error"
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.request(method, url, headers=headers, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:  # network / 5xx / timeout
                response = getattr(exc, "response", None)
                if response is not None:
                    text = response.text.replace("\n", " ")[:500]
                    last_error = f"{response.status_code} {response.reason}: {text}"
                else:
                    last_error = str(exc)
                if attempt >= self.config.max_retries:
                    break
                time.sleep(self.config.poll_interval_sec * attempt)
        raise Hunyuan3DError(f"{method} {url} failed after {self.config.max_retries} tries: {last_error}")

    # == authentication (Tencent OpenAI-compatible: Bearer key) ==========
    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    # == submit a reconstruction job =====================================
    def _submit_payload(self, image_b64: str, image_field: str) -> dict[str, Any]:
        """Build a TokenHub submit payload.

        TokenHub's 3D docs say native params are converted for the OpenAI-style
        endpoint, but the image field spelling has changed across examples and
        gateway versions. We try the documented snake_case first, then retry with
        lower-camel imageBase64 if the gateway says the image/prompt is empty.
        """
        payload: dict[str, Any] = {image_field: image_b64}
        if self.config.model:
            payload["model"] = self.config.model
        if self.config.enable_pbr is not None:
            payload["enable_pbr"] = self.config.enable_pbr
        if self.config.face_count is not None:
            payload["face_count"] = self.config.face_count
        if self.config.generate_type:
            payload["generate_type"] = self.config.generate_type
        if self.config.result_format:
            payload["result_format"] = self.config.result_format
        payload.update(self.config.extra_params)
        return payload

    def _submit_payload_once(self, payload: dict[str, Any]) -> str:
        resp = self._request("POST", "/v1/api/3d/submit", json=payload)
        data = _safe_json(resp)
        status = str(data.get("status", "")).lower()
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise Hunyuan3DSubmitError(data)
        job_id = data.get("id") or data.get("JobId") or data.get("job_id")
        if not job_id:
            raise Hunyuan3DError(f"submit_job: no job id in response: {data}")
        return str(job_id)

    @staticmethod
    def _looks_like_empty_image_error(exc: Hunyuan3DSubmitError) -> bool:
        error = exc.data.get("error") if isinstance(exc.data.get("error"), dict) else {}
        text = " ".join(str(error.get(k, "")) for k in ("message", "message_zh", "code"))
        return "ImageBase64" in text and ("为空" in text or "empty" in text.lower())

    def submit_job(self, image_path: Path) -> str:
        """Base64-encode the cutout, POST to /v1/api/3d/submit, return job id."""
        import base64

        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        try:
            return self._submit_payload_once(self._submit_payload(image_b64, "image_base64"))
        except Hunyuan3DSubmitError as exc:
            if not self._looks_like_empty_image_error(exc):
                raise
        return self._submit_payload_once(self._submit_payload(image_b64, "imageBase64"))

    # == poll job status =================================================
    def poll_job(self, task_id: str) -> dict[str, Any]:
        """POST /v1/api/3d/query {model, id} until status == completed."""
        body: dict[str, Any] = {"id": task_id}
        if self.config.model:
            body["model"] = self.config.model

        deadline = time.monotonic() + self.config.poll_timeout_sec
        while True:
            resp = self._request("POST", "/v1/api/3d/query", json=body)
            payload = _safe_json(resp)
            status = str(payload.get("status", "")).lower()
            if status in {"completed", "success", "succeeded", "done", "finished"}:
                return payload
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise Hunyuan3DError(f"Hunyuan3D job {task_id} failed: {payload}")
            # queued / in_progress -> keep waiting
            if time.monotonic() > deadline:
                raise Hunyuan3DError(f"Hunyuan3D job {task_id} timed out (status={status!r}).")
            time.sleep(self.config.poll_interval_sec)

    # == locate the mesh in the finished payload =========================
    @staticmethod
    def mesh_url_from_status(payload: dict[str, Any], prefer: str = "glb") -> tuple[str, str]:
        """Return (url, format) from the `data` array; prefer glb over obj.

        Completed query response carries `data: [{type, url, preview_image_url}]`.
        """
        entries = payload.get("data") or payload.get("result") or []
        if isinstance(entries, dict):
            entries = [entries]
        meshes = [
            e for e in entries
            if isinstance(e, dict) and e.get("url")
            and str(e.get("type", "")).lower() in {"glb", "obj", "stl", "usdz", "fbx", "ply"}
        ]
        if not meshes:
            raise Hunyuan3DError(f"No mesh url in completed payload: {payload}")
        meshes.sort(key=lambda e: 0 if str(e.get("type", "")).lower() == prefer else 1)
        chosen = meshes[0]
        return str(chosen["url"]), str(chosen.get("type", prefer)).lower()

    # -- download (safe to leave as-is) -----------------------------------

    def download_mesh(self, url: str, dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        resp = self._request("GET", url, stream=True)
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        if dest_path.stat().st_size == 0:
            raise Hunyuan3DError(f"Downloaded empty mesh from {url}")
        return dest_path

    # -- orchestration ----------------------------------------------------

    def reconstruct(self, image_path: Path, dest_path: Path) -> Hunyuan3DResult:
        """Full image -> mesh: submit, poll, download. Returns a Hunyuan3DResult.

        `dest_path` sets the on-disk stem/dir; the extension is rewritten to
        match whatever format the API actually returned (glb / obj / ...).
        """
        self.config.require_configured()
        image_path = Path(image_path)
        if not image_path.exists():
            raise Hunyuan3DError(f"Input image not found: {image_path}")
        task_id = self.submit_job(image_path)
        payload = self.poll_job(task_id)
        mesh_url, mesh_format = self.mesh_url_from_status(payload)
        dest_path = Path(dest_path).with_suffix(f".{mesh_format}")
        mesh_path = self.download_mesh(mesh_url, dest_path)
        return Hunyuan3DResult(
            mesh_path=mesh_path,
            task_id=task_id,
            mesh_format=mesh_format,
            raw_response=payload,
        )


def _safe_json(resp) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError as exc:
        raise Hunyuan3DError(f"Non-JSON response ({resp.status_code}): {resp.text[:300]}") from exc
    if not isinstance(data, dict):
        raise Hunyuan3DError(f"Expected a JSON object, got {type(data).__name__}: {data!r}")
    return data
