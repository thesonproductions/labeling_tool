"""Fixed-location folder import API for Label Studio.

Supports importing large image folders (2,000-3,000+ images) in sequential batches
and optional YOLO folder structures (`images/`, `labels/`, `classes.txt`) without
requiring host-level Local Files Storage bind mounts or Docker env changes.

Files are saved to a fixed directory per project and folder name:
    <MEDIA_ROOT>/<UPLOAD_DIR>/<project_id>/folders/<folder_name>/<filename>
and served natively through Label Studio's authenticated `/data/upload/...` endpoint.
"""

import hashlib
import io
import logging
import os
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.permissions import all_permissions
from core.utils.common import load_func
from core.utils.params import bool_from_request
from django.conf import settings
from django.db import transaction
from django.db.models import Max
from PIL import Image, UnidentifiedImageError
from projects.models import Project
from rest_framework import generics, status
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.settings import api_settings
from rest_framework.views import APIView
from tasks.functions import update_tasks_counters
from tasks.models import Annotation, Prediction, Task

from .models import FileUpload

logger = logging.getLogger(__name__)

ProjectImportPermission = load_func(settings.PROJECT_IMPORT_PERMISSION)

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
YOLO_META_DIR = '_yolo_meta'
_INVALID_PATH_CHARS_RE = re.compile(r'[\x00-\x1f<>:"|?*]')
_WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
}


def _validate_single_segment(raw_name: str, field_label: str) -> str:
    """Validate and normalize a single path segment (no traversal or path separators)."""
    if not isinstance(raw_name, str):
        raise ValidationError(f'Invalid {field_label}.')
    cleaned = raw_name.strip()
    if not cleaned or cleaned in ('.', '..'):
        raise ValidationError(f'Invalid {field_label}: "{raw_name}".')
    if '/' in cleaned or '\\' in cleaned or '..' in cleaned or '\x00' in cleaned:
        raise ValidationError(f'Path traversal is not allowed in {field_label}: "{raw_name}".')
    cleaned = _INVALID_PATH_CHARS_RE.sub('_', cleaned).strip(' .')
    if not cleaned or cleaned in ('.', '..'):
        raise ValidationError(f'Invalid {field_label}: "{raw_name}".')
    if cleaned.split('.', 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = f'_{cleaned}'
    return cleaned


def _extract_safe_filename(uploaded_name: str) -> str:
    """Extract a safe basename from a browser-supplied upload filename (rejecting traversal)."""
    if not isinstance(uploaded_name, str) or '\x00' in uploaded_name:
        raise ValidationError('Invalid uploaded file name.')
    normalized = uploaded_name.replace('\\', '/').strip()
    parts = [p for p in normalized.split('/') if p]
    if not parts or any(p in ('.', '..') for p in parts):
        raise ValidationError(f'Unsafe file path: "{uploaded_name}".')
    return _validate_single_segment(parts[-1], 'file name')


def _ensure_within_dir(base_dir: Path, target_path: Path) -> Path:
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()
    if resolved_target != resolved_base and resolved_base not in resolved_target.parents:
        raise ValidationError('Target path escapes allowed project storage directory.')
    return resolved_target


def _get_folder_storage_paths(project_id: int, folder_name: str) -> Tuple[str, Path]:
    safe_folder = _validate_single_segment(folder_name, 'folder_name')
    upload_prefix = settings.UPLOAD_DIR.strip('/\\')
    rel_folder_key = f'{upload_prefix}/{project_id}/folders/{safe_folder}'
    media_root = Path(settings.MEDIA_ROOT)
    base_project_folders = media_root / upload_prefix / str(project_id) / 'folders'
    disk_folder = base_project_folders / safe_folder
    _ensure_within_dir(base_project_folders, disk_folder)
    return rel_folder_key, disk_folder


def _inspect_project_image_config(project: Project) -> Tuple[str, str, str]:
    """Return (image_data_key, to_name, from_name) from the project's label config."""
    image_key = 'image'
    to_name = 'image'
    from_name = 'label'
    try:
        parsed = project.get_parsed_config() or {}
        for tag_name, cfg in parsed.items():
            tag_type = str(cfg.get('type', '')).lower()
            if tag_type == 'rectanglelabels':
                from_name = tag_name
                to_names = cfg.get('to_name') or []
                if to_names:
                    to_name = to_names[0]
                inputs = cfg.get('inputs') or []
                for inp in inputs:
                    if str(inp.get('type', '')).lower() == 'image' and inp.get('value'):
                        image_key = str(inp['value']).lstrip('$')
                        break
                break
        else:
            # Fallback: look for any Image input tag (e.g., PolygonLabels-only configs)
            for tag_name, cfg in parsed.items():
                inputs = cfg.get('inputs') or []
                for inp in inputs:
                    if str(inp.get('type', '')).lower() == 'image' and inp.get('value'):
                        image_key = str(inp['value']).lstrip('$')
                        to_names = cfg.get('to_name') or []
                        if to_names:
                            to_name = to_names[0]
                        break
    except Exception:
        logger.debug('Falling back to default image/label config names for project %s', project.id)
    return image_key, to_name, from_name


def _parse_yolo_label_file(
    txt_path: Path,
    img_w: int,
    img_h: int,
    classes: List[str],
    from_name: str,
    to_name: str,
) -> List[Dict[str, Any]]:
    if not txt_path.is_file():
        return []
    regions: List[Dict[str, Any]] = []
    try:
        lines = txt_path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
        except ValueError:
            continue

        x_pct = max(0.0, min(100.0, (xc - w_norm / 2.0) * 100.0))
        y_pct = max(0.0, min(100.0, (yc - h_norm / 2.0) * 100.0))
        w_pct = max(0.0, min(100.0 - x_pct, w_norm * 100.0))
        h_pct = max(0.0, min(100.0 - y_pct, h_norm * 100.0))
        if w_pct <= 0.0 or h_pct <= 0.0:
            continue

        label_name = classes[class_id] if 0 <= class_id < len(classes) else f'class_{class_id}'
        regions.append(
            {
                'id': f'box_{uuid.uuid4().hex[:8]}',
                'from_name': from_name,
                'to_name': to_name,
                'type': 'rectanglelabels',
                'original_width': img_w,
                'original_height': img_h,
                'image_rotation': 0,
                'origin': 'manual',
                'value': {
                    'x': round(x_pct, 4),
                    'y': round(y_pct, 4),
                    'width': round(w_pct, 4),
                    'height': round(h_pct, 4),
                    'rotation': 0,
                    'rectanglelabels': [label_name],
                },
            }
        )
    return regions


_LABEL_COLOR_PALETTE = (
    '#27AE60', '#2980B9', '#E67E22', '#8E44AD', '#E74C3C', '#16A085',
    '#D35400', '#2C3E50', '#C0392B', '#1ABC9C', '#F39C12', '#7F8C8D',
    '#3498DB', '#9B59B6', '#2ECC71', '#E84393', '#00B894', '#0984E3',
    '#6C5CE7', '#FD79A8',
)
_SYNCABLE_LABEL_TAGS = {'RectangleLabels', 'PolygonLabels'}


def _pick_label_color(label_name: str) -> str:
    digest = hashlib.md5(label_name.encode('utf-8')).digest()
    idx = int.from_bytes(digest[:2], 'big') % len(_LABEL_COLOR_PALETTE)
    return _LABEL_COLOR_PALETTE[idx]


def _sync_classes_into_label_config(project: Project, classes_list: List[str]) -> int:
    """Append missing classes from classes.txt into RectangleLabels/PolygonLabels in project.label_config."""
    if not classes_list or not project.label_config:
        return 0

    try:
        root = ET.fromstring(project.label_config)
    except ET.ParseError:
        logger.debug('Skipping label_config sync for project %s: invalid XML', project.id)
        return 0

    added_unique = set()
    modified = False
    for elem in root.iter():
        if elem.tag not in _SYNCABLE_LABEL_TAGS:
            continue
        existing = {
            child.attrib.get('value')
            for child in elem
            if child.tag == 'Label' and child.attrib.get('value') is not None
        }
        for cls_name in classes_list:
            if cls_name in existing:
                continue
            label_el = ET.SubElement(elem, 'Label', {'value': cls_name, 'background': _pick_label_color(cls_name)})
            label_el.tail = '\n'
            existing.add(cls_name)
            added_unique.add(cls_name)
            modified = True

    if not modified:
        return 0

    new_config = ET.tostring(root, encoding='unicode')
    try:
        project.validate_config(new_config)
        project.label_config = new_config
        project.save(update_fields=['label_config'])
        return len(added_unique)
    except Exception:
        logger.warning('Failed to sync classes.txt into project %s label_config', project.id, exc_info=True)
        return 0


def _collect_uploaded_files(request, field_names: Tuple[str, ...] = ('files', 'file')) -> List[Any]:
    collected: List[Any] = []
    seen_keys = set()
    for fname in field_names:
        if fname in request.FILES:
            collected.extend(request.FILES.getlist(fname))
            seen_keys.add(fname)
    for key in request.FILES:
        if key not in seen_keys:
            collected.extend(request.FILES.getlist(key))
    return collected


class ImportFolderAPI(APIView):
    """Upload a local image or YOLO folder in batches into a fixed project folder."""

    permission_required = all_permissions.projects_change
    permission_classes = api_settings.DEFAULT_PERMISSION_CLASSES + [ProjectImportPermission]
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, pk: int, *args, **kwargs):
        project = generics.get_object_or_404(Project.objects.for_user(request.user), pk=pk)
        folder_name = request.data.get('folder_name') or 'imported_folder'
        mode = str(request.data.get('mode') or 'images').strip().lower()

        rel_folder_key, disk_folder = _get_folder_storage_paths(project.id, folder_name)
        disk_folder.mkdir(parents=True, exist_ok=True)
        meta_dir = disk_folder / YOLO_META_DIR
        labels_dir = meta_dir / 'labels'

        if mode == 'yolo_meta':
            meta_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            classes_count = 0
            labels_synced = 0
            classes_file = request.FILES.get('classes')
            if classes_file is not None:
                raw_classes = classes_file.read()
                classes_path = _ensure_within_dir(meta_dir, meta_dir / 'classes.txt')
                classes_path.write_bytes(raw_classes)
                classes_list = [
                    ln.strip() for ln in raw_classes.decode('utf-8', errors='replace').splitlines() if ln.strip()
                ]
                classes_count = len(classes_list)
                labels_synced = _sync_classes_into_label_config(project, classes_list)

            labels_saved = 0
            for label_file in request.FILES.getlist('labels'):
                safe_lbl_name = _extract_safe_filename(label_file.name)
                if not safe_lbl_name.lower().endswith('.txt'):
                    continue
                lbl_path = _ensure_within_dir(labels_dir, labels_dir / safe_lbl_name)
                lbl_path.write_bytes(label_file.read())
                labels_saved += 1

            return Response(
                {
                    'folder_name': Path(rel_folder_key).name,
                    'storage_path': rel_folder_key,
                    'classes_count': classes_count,
                    'labels_synced': labels_synced,
                    'labels_saved': labels_saved,
                },
                status=status.HTTP_200_OK,
            )

        if mode != 'images':
            raise ValidationError(f'Unsupported import mode: "{mode}". Expected "images" or "yolo_meta".')

        skip_existing = bool_from_request(request.data, 'skip_existing', default=True)
        yolo_mode = str(request.data.get('yolo_mode') or 'annotations').strip().lower()
        if yolo_mode not in ('annotations', 'predictions', 'none'):
            raise ValidationError('yolo_mode must be one of: annotations, predictions, none.')

        uploaded_files = _collect_uploaded_files(request, field_names=('files', 'file'))
        if not uploaded_files:
            raise ValidationError('No image files were provided.')

        classes: List[str] = []
        classes_path = meta_dir / 'classes.txt'
        if classes_path.is_file():
            classes = [
                ln.strip()
                for ln in classes_path.read_text(encoding='utf-8', errors='replace').splitlines()
                if ln.strip()
            ]

        image_key, to_name, from_name = _inspect_project_image_config(project)

        created_tasks: List[Task] = []
        annotations_created = 0
        predictions_created = 0
        skipped_existing = 0
        skipped_invalid: List[str] = []

        with transaction.atomic():
            next_inner_id = (project.tasks.aggregate(Max('inner_id')).get('inner_id__max') or 0) + 1

            for uploaded in uploaded_files:
                safe_name = _extract_safe_filename(uploaded.name)
                ext = os.path.splitext(safe_name)[1].lower()
                if ext not in ALLOWED_IMAGE_EXTENSIONS:
                    skipped_invalid.append(safe_name)
                    continue

                raw_bytes = uploaded.read()
                if not raw_bytes:
                    skipped_invalid.append(safe_name)
                    continue

                try:
                    with Image.open(io.BytesIO(raw_bytes)) as im:
                        orig_w, orig_h = im.size
                        im.verify()
                    if orig_w <= 0 or orig_h <= 0:
                        skipped_invalid.append(safe_name)
                        continue
                except (UnidentifiedImageError, OSError, ValueError):
                    skipped_invalid.append(safe_name)
                    continue

                rel_file_key = f'{rel_folder_key}/{safe_name}'
                if skip_existing and Task.objects.filter(project=project, file_upload__file=rel_file_key).exists():
                    skipped_existing += 1
                    continue

                disk_file_path = _ensure_within_dir(disk_folder, disk_folder / safe_name)
                disk_file_path.write_bytes(raw_bytes)

                file_upload = FileUpload.objects.filter(project=project, file=rel_file_key).last()
                if file_upload is None:
                    file_upload = FileUpload(user=request.user, project=project)
                    file_upload.file.name = rel_file_key
                    file_upload.save()

                task = Task.objects.create(
                    project=project,
                    data={image_key: file_upload.url},
                    file_upload=file_upload,
                    inner_id=next_inner_id,
                )
                next_inner_id += 1
                created_tasks.append(task)

                if yolo_mode != 'none' and labels_dir.is_dir():
                    stem = os.path.splitext(safe_name)[0]
                    lbl_file = labels_dir / f'{stem}.txt'
                    if lbl_file.is_file():
                        regions = _parse_yolo_label_file(
                            lbl_file,
                            img_w=orig_w,
                            img_h=orig_h,
                            classes=classes,
                            from_name=from_name,
                            to_name=to_name,
                        )
                        if regions:
                            if yolo_mode == 'predictions':
                                Prediction.objects.create(
                                    task=task,
                                    project=project,
                                    result=regions,
                                    model_version='yolo_folder_import',
                                )
                                predictions_created += 1
                            else:
                                Annotation.objects.create(
                                    task=task,
                                    project=project,
                                    completed_by=request.user,
                                    result=regions,
                                )
                                annotations_created += 1

        if created_tasks:
            try:
                update_tasks_counters(created_tasks)
            except Exception as exc:
                logger.warning('update_tasks_counters skipped: %s', exc)
            try:
                project.summary.update_data_columns(created_tasks)
            except Exception as exc:
                logger.warning('project.summary.update_data_columns skipped: %s', exc)

        return Response(
            {
                'folder_name': Path(rel_folder_key).name,
                'storage_path': rel_folder_key,
                'task_count': len(created_tasks),
                'annotations_created': annotations_created,
                'predictions_created': predictions_created,
                'skipped_existing': skipped_existing,
                'skipped_invalid': skipped_invalid,
                'task_ids': [t.id for t in created_tasks],
            },
            status=status.HTTP_201_CREATED,
        )
