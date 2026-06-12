"""Pydantic schemas for the wizard API."""

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class PipelineType(str, Enum):
    detector_ocr = "detector_ocr"
    qr_scanner = "qr_scanner"


class PackagingCreate(BaseModel):
    key: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$", max_length=50)
    display_name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    pipeline: PipelineType = PipelineType.detector_ocr
    sub_regions: list[str] = Field(default_factory=lambda: ["lot"])

    @field_validator("sub_regions")
    @classmethod
    def _validate_sub_regions(cls, v: list[str]) -> list[str]:
        if not v:
            return ["lot"]
        cleaned = []
        for r in v:
            r = r.strip().lower()
            if not r:
                continue
            if not all(c.isalnum() or c == "_" for c in r):
                raise ValueError(f"invalid sub-region '{r}' — ใช้ a-z, 0-9, _ เท่านั้น")
            cleaned.append(r)
        if not cleaned:
            return ["lot"]
        return cleaned


class PackagingUpdate(BaseModel):
    display_name: str | None = None
    description: str | None = None
    pipeline: PipelineType | None = None


class PackagingConfigUpdate(BaseModel):
    lot_patterns: list[str] = Field(default_factory=list)
    fields_extracted: list[str] = Field(default_factory=lambda: ["lot"])
    sheet_checks: list[str] = Field(default_factory=list)
    message_template_key: str = "default_full"

    @field_validator("lot_patterns")
    @classmethod
    def _validate_regex(cls, v: list[str]) -> list[str]:
        for p in v:
            try:
                re.compile(p)
            except re.error as e:
                raise ValueError(f"invalid regex {p!r}: {e}")
        return v


class PackagingResponse(BaseModel):
    key: str
    display_name: str
    description: str | None = None
    pipeline: str
    status: str
    image_count: int = 0
    conf_threshold: float | None = None
    accuracy: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    config: dict | None = None


class ConfThresholdUpdate(BaseModel):
    """Runtime tuning ของ active packaging (ADR 0004).

    ต่ำกว่า 0.5 → รูปที่ classifier เดามั่วหลุดเข้า pipeline ผิดประเภท,
    สูงกว่า 0.95 → class โดนปัดแทบทุกรูป
    """
    conf_threshold: float = Field(..., ge=0.5, le=0.95)


class ConfThresholdResponse(BaseModel):
    key: str
    conf_threshold: float
    previous: float


class RegexPreviewRequest(BaseModel):
    examples: list[str] = Field(..., min_length=1, max_length=20)


class RegexMatch(BaseModel):
    input: str
    match: str | None
    ok: bool


class RegexPreviewResponse(BaseModel):
    pattern: str
    matches: list[RegexMatch]


class BboxAnnotation(BaseModel):
    """Single bbox — pixel coords on original image."""
    x1: float = Field(..., ge=0)
    y1: float = Field(..., ge=0)
    x2: float = Field(..., ge=0)
    y2: float = Field(..., ge=0)
    label: str | None = None  # 'box' / 'sachet' / None

    @field_validator("x2")
    @classmethod
    def _x2_gt_x1(cls, v, info):
        x1 = info.data.get("x1", 0)
        if v <= x1:
            raise ValueError(f"x2 ({v}) must be > x1 ({x1})")
        return v

    @field_validator("y2")
    @classmethod
    def _y2_gt_y1(cls, v, info):
        y1 = info.data.get("y1", 0)
        if v <= y1:
            raise ValueError(f"y2 ({v}) must be > y1 ({y1})")
        return v


class AnnotationSave(BaseModel):
    bboxes: list[BboxAnnotation]
