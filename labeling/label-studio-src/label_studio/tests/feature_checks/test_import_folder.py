import io
import json

import pytest
from data_import.folder_import import _extract_safe_filename
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from PIL import Image
from projects.models import Project
from rest_framework.exceptions import ValidationError
from tasks.models import Annotation, Prediction, Task


def _make_jpeg_bytes(size=(100, 100), color=(60, 140, 80)):
    buf = io.BytesIO()
    Image.new('RGB', size, color=color).save(buf, format='JPEG')
    return buf.getvalue()


def _create_detection_project(business_client, title='Import Folder Check'):
    label_config = """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="Rầy nâu"/>
        <Label value="Philonthus"/>
      </RectangleLabels>
    </View>
    """
    resp = business_client.post(
        '/api/projects/',
        data=json.dumps({'title': title, 'label_config': label_config}),
        content_type='application/json',
    )
    assert resp.status_code == 201, resp.content
    return resp.json()['id']


@pytest.mark.django_db
def test_import_folder_batches_saves_to_fixed_directory_and_serves_images(business_client):
    project_id = _create_detection_project(business_client, title='Fixed Folder Batch')
    img_bytes = _make_jpeg_bytes()

    for batch_idx in range(2):
        batch_files = [
            SimpleUploadedFile(f'trap_{batch_idx}_{i}.jpg', img_bytes, content_type='image/jpeg')
            for i in range(3)
        ]
        resp = business_client.post(
            f'/api/projects/{project_id}/import-folder',
            data={
                'mode': 'images',
                'folder_name': 'Raw_2026',
                'skip_existing': 'true',
                'files': batch_files,
            },
        )
        assert resp.status_code == 201, resp.content
        payload = resp.json()
        assert payload['task_count'] == 3
        assert payload['storage_path'] == f'upload/{project_id}/folders/Raw_2026'

    tasks = list(Task.objects.filter(project_id=project_id).order_by('id'))
    assert len(tasks) == 6
    expected_url = f'/data/upload/{project_id}/folders/Raw_2026/trap_0_0.jpg'
    assert tasks[0].data['image'].endswith(expected_url)

    # Verify uploaded file is actually served back via Label Studio's /data/upload/ endpoint
    get_resp = business_client.get(expected_url)
    assert get_resp.status_code == 200


@pytest.mark.django_db
def test_import_folder_yolo_structure_creates_annotations_and_predictions(business_client):
    project_id = _create_detection_project(business_client, title='YOLO Folder Import')
    img_bytes = _make_jpeg_bytes(size=(200, 100))

    classes_file = SimpleUploadedFile('classes.txt', 'Rầy nâu\nPhilonthus\n'.encode('utf-8'), content_type='text/plain')
    lbl_1 = SimpleUploadedFile('insect_01.txt', b'0 0.5 0.5 0.2 0.2\n', content_type='text/plain')
    lbl_2 = SimpleUploadedFile('insect_02.txt', b'1 0.25 0.5 0.1 0.2\n', content_type='text/plain')

    meta_resp = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'yolo_meta',
            'folder_name': 'Raw',
            'classes': classes_file,
            'labels': [lbl_1, lbl_2],
        },
    )
    assert meta_resp.status_code == 200, meta_resp.content
    assert meta_resp.json()['classes_count'] == 2
    assert meta_resp.json()['labels_saved'] == 2

    # Import insect_01 as Annotation
    ann_resp = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'images',
            'folder_name': 'Raw',
            'yolo_mode': 'annotations',
            'files': [SimpleUploadedFile('insect_01.jpg', img_bytes, content_type='image/jpeg')],
        },
    )
    assert ann_resp.status_code == 201, ann_resp.content
    assert ann_resp.json()['annotations_created'] == 1

    ann = Annotation.objects.filter(project_id=project_id).first()
    assert ann is not None
    assert len(ann.result) == 1
    assert ann.result[0]['value']['rectanglelabels'] == ['Rầy nâu']
    assert ann.result[0]['value']['x'] == 40.0
    assert ann.result[0]['value']['y'] == 40.0
    assert ann.result[0]['value']['width'] == 20.0
    assert ann.result[0]['value']['height'] == 20.0

    # Import insect_02 as Prediction
    pred_resp = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'images',
            'folder_name': 'Raw',
            'yolo_mode': 'predictions',
            'files': [SimpleUploadedFile('insect_02.jpg', img_bytes, content_type='image/jpeg')],
        },
    )
    assert pred_resp.status_code == 201, pred_resp.content
    assert pred_resp.json()['predictions_created'] == 1
    pred = Prediction.objects.filter(project_id=project_id).first()
    assert pred is not None
    assert pred.result[0]['value']['rectanglelabels'] == ['Philonthus']


@pytest.mark.django_db
def test_import_folder_skips_zero_byte_and_corrupt_files_without_failing_batch(business_client):
    project_id = _create_detection_project(business_client, title='Skip Invalid Files')

    resp = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'images',
            'folder_name': 'Mixed_Batch',
            'files': [
                SimpleUploadedFile('valid.jpg', _make_jpeg_bytes(), content_type='image/jpeg'),
                SimpleUploadedFile('empty_0b.jpg', b'', content_type='image/jpeg'),
                SimpleUploadedFile('corrupt.jpg', b'not-an-image', content_type='image/jpeg'),
                SimpleUploadedFile('README.md', b'# notes', content_type='text/markdown'),
            ],
        },
    )
    assert resp.status_code == 201, resp.content
    payload = resp.json()
    assert payload['task_count'] == 1
    assert set(payload['skipped_invalid']) == {'empty_0b.jpg', 'corrupt.jpg', 'README.md'}
    assert Task.objects.filter(project_id=project_id).count() == 1


@pytest.mark.django_db
def test_import_folder_skip_existing_prevents_duplicates_on_resume(business_client):
    project_id = _create_detection_project(business_client, title='Resume Deduplication')
    img_bytes = _make_jpeg_bytes()

    r1 = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'images',
            'folder_name': 'Resume_Set',
            'skip_existing': 'true',
            'files': [
                SimpleUploadedFile('img_1.jpg', img_bytes, content_type='image/jpeg'),
                SimpleUploadedFile('img_2.jpg', img_bytes, content_type='image/jpeg'),
            ],
        },
    )
    assert r1.status_code == 201, r1.content
    assert r1.json()['task_count'] == 2

    # Re-upload overlapping batch as if resuming after disconnect
    r2 = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'images',
            'folder_name': 'Resume_Set',
            'skip_existing': 'true',
            'files': [
                SimpleUploadedFile('img_1.jpg', img_bytes, content_type='image/jpeg'),
                SimpleUploadedFile('img_2.jpg', img_bytes, content_type='image/jpeg'),
                SimpleUploadedFile('img_3.jpg', img_bytes, content_type='image/jpeg'),
            ],
        },
    )
    assert r2.status_code == 201, r2.content
    assert r2.json()['task_count'] == 1
    assert r2.json()['skipped_existing'] == 2
    assert Task.objects.filter(project_id=project_id).count() == 3


@pytest.mark.django_db
def test_import_folder_blocks_path_traversal_and_unauthenticated(business_client):
    project_id = _create_detection_project(business_client, title='Folder Security')
    img_bytes = _make_jpeg_bytes()

    # Path traversal or nested separators in folder_name must be rejected
    for bad_folder_name in ('../../outside', 'nested/sub', '..'):
        bad_folder = business_client.post(
            f'/api/projects/{project_id}/import-folder',
            data={
                'mode': 'images',
                'folder_name': bad_folder_name,
                'files': [SimpleUploadedFile('a.jpg', img_bytes, content_type='image/jpeg')],
            },
        )
        assert bad_folder.status_code == 400, bad_folder.content

    # Path traversal helper rejects unsafe filenames directly
    with pytest.raises(ValidationError):
        _extract_safe_filename('../evil.jpg')

    # Windows reserved device names must be safely prefixed on all platforms
    assert _extract_safe_filename('CON.jpg') == '_CON.jpg'
    assert _extract_safe_filename('nul.png') == '_nul.png'
    res_reserved = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'images',
            'folder_name': 'CON',
            'files': [SimpleUploadedFile('AUX.jpg', img_bytes, content_type='image/jpeg')],
        },
    )
    assert res_reserved.status_code == 201, res_reserved.content
    assert res_reserved.json()['folder_name'] == '_CON'
    assert res_reserved.json()['task_count'] == 1
    Task.objects.filter(project_id=project_id).delete()

    # Unauthenticated request must be rejected
    anon = Client()
    unauth = anon.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'images',
            'folder_name': 'SafeFolder',
            'files': [SimpleUploadedFile('a.jpg', img_bytes, content_type='image/jpeg')],
        },
    )
    assert unauth.status_code in (401, 403), unauth.content
    assert Task.objects.filter(project_id=project_id).count() == 0


@pytest.mark.django_db
def test_uploaded_file_thumbnail_and_http_caching(business_client):
    project_id = _create_detection_project(business_client, title='Thumbnail & Cache Check')
    large_img_bytes = _make_jpeg_bytes(size=(1200, 900))

    resp = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'images',
            'folder_name': 'Thumb_Set',
            'files': [SimpleUploadedFile('dish_01.jpg', large_img_bytes, content_type='image/jpeg')],
        },
    )
    assert resp.status_code == 201, resp.content
    img_url = f'/data/upload/{project_id}/folders/Thumb_Set/dish_01.jpg'

    # Full image includes private Cache-Control & ETag, and supports 304 Not Modified
    full_resp = business_client.get(img_url)
    assert full_resp.status_code == 200
    assert full_resp['Cache-Control'] == 'private, max-age=86400'
    full_etag = full_resp['ETag']
    assert full_etag

    full_304 = business_client.get(img_url, HTTP_IF_NONE_MATCH=full_etag)
    assert full_304.status_code == 304
    assert full_304['ETag'] == full_etag

    # Thumbnail request returns resized JPEG <= requested width with its own ETag
    thumb_resp = business_client.get(f'{img_url}?thumb=1&w=160')
    assert thumb_resp.status_code == 200
    assert thumb_resp['Content-Type'] == 'image/jpeg'
    assert thumb_resp['Cache-Control'] == 'private, max-age=86400'
    thumb_etag = thumb_resp['ETag']
    assert thumb_etag and thumb_etag != full_etag

    thumb_bytes = b''.join(thumb_resp.streaming_content)
    with Image.open(io.BytesIO(thumb_bytes)) as thumb_im:
        assert max(thumb_im.size) <= 160

    thumb_304 = business_client.get(f'{img_url}?thumb=1&w=160', HTTP_IF_NONE_MATCH=thumb_etag)
    assert thumb_304.status_code == 304
    assert thumb_304['ETag'] == thumb_etag


@pytest.mark.django_db
def test_import_folder_yolo_meta_syncs_missing_classes_to_label_config(business_client):
    project_id = _create_detection_project(business_client, title='Sync Classes Check')
    classes_text = 'Rầy nâu\nPhilonthus\nBọ cánh cứng Lithocharodes longicollis\nBọ nước Hydrophilus spp.\n'
    classes_file = SimpleUploadedFile('classes.txt', classes_text.encode('utf-8'), content_type='text/plain')

    meta_resp = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'yolo_meta',
            'folder_name': 'Raw_Sync',
            'classes': classes_file,
        },
    )
    assert meta_resp.status_code == 200, meta_resp.content
    payload = meta_resp.json()
    assert payload['classes_count'] == 4
    assert payload['labels_synced'] == 2

    project = Project.objects.get(pk=project_id)
    assert 'Bọ cánh cứng Lithocharodes longicollis' in project.label_config
    assert 'Bọ nước Hydrophilus spp.' in project.label_config
    assert project.label_config.count('value="Rầy nâu"') == 1

    # Re-sending identical classes.txt is idempotent (0 newly synced)
    classes_file_2 = SimpleUploadedFile('classes.txt', classes_text.encode('utf-8'), content_type='text/plain')
    meta_resp_2 = business_client.post(
        f'/api/projects/{project_id}/import-folder',
        data={
            'mode': 'yolo_meta',
            'folder_name': 'Raw_Sync',
            'classes': classes_file_2,
        },
    )
    assert meta_resp_2.status_code == 200, meta_resp_2.content
    assert meta_resp_2.json()['labels_synced'] == 0
