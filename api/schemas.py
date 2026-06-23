"""Pydantic schemas for the wizard API."""

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from utils.field_groups import validate_groups


class PipelineType(str, Enum):
    detector_ocr = "detector_ocr"
    qr_scanner = "qr_scanner"


class DetectionMode(str, Enum):
    single = "single"
    cross_check = "cross_check"
    multi_field = "multi_field"


class PackagingCreate(BaseModel):
    key: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$", max_length=50)
    display_name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    pipeline: PipelineType = PipelineType.detector_ocr
    sub_regions: list[str] = Field(default_factory=lambda: ["lot"])
    detection_mode: DetectionMode = DetectionMode.single

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

    @model_validator(mode="after")
    def _check_detection_mode(self):
        if self.detection_mode == DetectionMode.multi_field:
            self.sub_regions = validate_groups(self.sub_regions)
        elif self.detection_mode == DetectionMode.cross_check:
            if len(self.sub_regions) < 2:
                raise ValueError("cross_check ต้องมีอย่างน้อย 2 sub-regions (เช่น box, sachet)")
        return self


class PackagingUpdate(BaseModel):
    display_name: str | None = None
    description: str | None = None
    pipeline: PipelineType | None = None


class ProductAlias(BaseModel):
    """keyword บนบรรจุภัณฑ์ → canonical name ที่ตรงกับคอลัมน์ Product Name ใน sheet."""
    canonical: str
    keywords: list[str] = Field(default_factory=list)


class PackagingConfigUpdate(BaseModel):
    lot_patterns: list[str] = Field(default_factory=list)
    fields_extracted: list[str] = Field(default_factory=lambda: ["lot"])
    sheet_checks: list[str] = Field(default_factory=list)
    message_template_key: str = "default_full"
    product_aliases: list[ProductAlias] = Field(default_factory=list)

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
    sub_regions: list[str] | None = None
    detection_mode: str | None = None
    product_aliases: list[ProductAlias] | None = None
    fields_extracted: list[str] | None = None
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


class ProductAliasesUpdate(BaseModel):
    """Runtime edit of the product names an active class reads (no retrain)."""
    product_aliases: list[ProductAlias] = Field(..., min_length=1)


class ProductAliasesResponse(BaseModel):
    key: str
    product_aliases: list[ProductAlias]
    previous: list[ProductAlias]


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
