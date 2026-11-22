import io
import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from PIL import Image
from tasks.models import Task


def _make_image_bytes(fmt='JPEG', size=(32, 32), color=(40, 120, 80)):
    buf = io.BytesIO()
    Image.new('RGB', size, color=color).save(buf, format=fmt)
    return buf.getvalue()


def _create_image_project(business_client, title='Image Import Checks'):
    label_config = """
    <View>
      <Image name="image" value="$image"/>
      <PolygonLabels name="labels" toName="image">
        <Label value="insect"/>
      </PolygonLabels>
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
def test_import_multiple_image_formats_via_multipart(business_client):
    project_id = _create_image_project(business_client, title='Multi Format Import')

    files = {
        'file_0': SimpleUploadedFile('sample_1.jpg', _make_image_bytes('JPEG'), content_type='image/jpeg'),
        'file_1': SimpleUploadedFile('sample_2.png', _make_image_bytes('PNG'), content_type='image/png'),
        'file_2': SimpleUploadedFile('sample_3.webp', _make_image_bytes('WEBP'), content_type='image/webp'),
    }
    resp = business_client.post(f'/api/projects/{project_id}/import', data=files)
    assert resp.status_code == 201, resp.content
    assert resp.json()['task_count'] == 3
    assert Task.objects.filter(project_id=project_id).count() == 3


@pytest.mark.django_db
def test_import_images_in_sequential_batches_accumulates_tasks(business_client):
    project_id = _create_image_project(business_client, title='Batch Import')
    jpg_bytes = _make_image_bytes('JPEG')

    for batch_idx in range(3):
        batch_files = {
            f'file_{i}': SimpleUploadedFile(
                f'batch_{batch_idx}_img_{i}.jpg', jpg_bytes, content_type='image/jpeg'
            )
            for i in range(5)
        }
        resp = business_client.post(f'/api/projects/{project_id}/import', data=batch_files)
        assert resp.status_code == 201, resp.content
        assert resp.json()['task_count'] == 5

    assert Task.objects.filter(project_id=project_id).count() == 15


@pytest.mark.django_db
def test_import_image_with_vietnamese_unicode_filename(business_client):
    project_id = _create_image_project(business_client, title='Unicode Filename Import')
    upload = SimpleUploadedFile(
        'Rầy nâu (bẫy 1).jpg',
        _make_image_bytes('JPEG'),
        content_type='image/jpeg',
    )
    resp = business_client.post(f'/api/projects/{project_id}/import', data={'file': upload})
    assert resp.status_code == 201, resp.content
    assert Task.objects.filter(project_id=project_id).count() == 1


@pytest.mark.django_db
def test_import_invalid_json_returns_error_and_preserves_existing_tasks(business_client):
    project_id = _create_image_project(business_client, title='Preserve Tasks On Error')

    valid_img = SimpleUploadedFile('keep_me.jpg', _make_image_bytes('JPEG'), content_type='image/jpeg')
    ok_resp = business_client.post(f'/api/projects/{project_id}/import', data={'file': valid_img})
    assert ok_resp.status_code == 201, ok_resp.content
    existing_ids = list(Task.objects.filter(project_id=project_id).values_list('id', flat=True))
    assert len(existing_ids) == 1

    bad_json = SimpleUploadedFile('broken.json', b'{not valid json', content_type='application/json')
    err_resp = business_client.post(f'/api/projects/{project_id}/import', data={'file': bad_json})
    assert err_resp.status_code == 400, err_resp.content

    remaining_ids = list(Task.objects.filter(project_id=project_id).values_list('id', flat=True))
    assert remaining_ids == existing_ids


@pytest.mark.django_db
def test_localfiles_relative_recursive_sync_is_idempotent(business_client, settings, tmp_path):
    doc_root = tmp_path / 'server_images'
    sub_a = doc_root / 'Raw' / 'dot_1'
    sub_b = doc_root / 'Raw' / 'dot_2'
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)

    (sub_a / 'img_a.jpg').write_bytes(_make_image_bytes('JPEG'))
    (sub_b / 'img_b.png').write_bytes(_make_image_bytes('PNG'))

    settings.LOCAL_FILES_SERVING_ENABLED = True
    settings.LOCAL_FILES_DOCUMENT_ROOT = str(doc_root)

    project_id = _create_image_project(business_client, title='Recursive Local Sync')
    create_resp = business_client.post(
        '/api/storages/localfiles/',
        data=json.dumps({
            'project': project_id,
            'title': 'Raw Nested',
            'path': 'Raw',
            'use_blob_urls': True,
            'recursive_scan': True,
        }),
        content_type='application/json',
    )
    assert create_resp.status_code == 201, create_resp.content
    storage_id = create_resp.json()['id']

    sync_1 = business_client.post(f'/api/storages/localfiles/{storage_id}/sync')
    assert sync_1.status_code == 200, sync_1.content
    assert Task.objects.filter(project_id=project_id).count() == 2

    # Second sync must not duplicate existing tasks
    sync_2 = business_client.post(f'/api/storages/localfiles/{storage_id}/sync')
    assert sync_2.status_code == 200, sync_2.content
    assert Task.objects.filter(project_id=project_id).count() == 2


@pytest.mark.django_db
def test_import_rejects_unauthenticated_request(business_client):
    project_id = _create_image_project(business_client, title='Auth Required Import')

    anon_client = Client()
    upload = SimpleUploadedFile('unauth.jpg', _make_image_bytes('JPEG'), content_type='image/jpeg')
    resp = anon_client.post(f'/api/projects/{project_id}/import', data={'file': upload})
    assert resp.status_code in (401, 403), resp.content
    assert Task.objects.filter(project_id=project_id).count() == 0
