import json
import os

import pytest
from tasks.models import Task


def _create_project(business_client):
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
        data=json.dumps({'title': 'Relative Storage Test', 'label_config': label_config}),
        content_type='application/json',
    )
    assert resp.status_code == 201, resp.content
    return resp.json()['id']


@pytest.mark.django_db
def test_localfiles_accepts_relative_path_and_syncs_images(business_client, settings, tmp_path):
    doc_root = tmp_path / 'files_root'
    sub_dir = doc_root / 'Raw' / 'images'
    sub_dir.mkdir(parents=True)
    (sub_dir / 'insect_01.jpg').write_bytes(b'\xff\xd8\xff\xd9')

    settings.LOCAL_FILES_SERVING_ENABLED = True
    settings.LOCAL_FILES_DOCUMENT_ROOT = str(doc_root)

    project_id = _create_project(business_client)
    resp = business_client.post(
        '/api/storages/localfiles/',
        data=json.dumps({
            'project': project_id,
            'title': 'Relative Source',
            'path': 'Raw/images',
            'use_blob_urls': True,
        }),
        content_type='application/json',
    )
    assert resp.status_code == 201, resp.content
    storage = resp.json()
    assert os.path.normpath(storage['path']) == os.path.normpath(str(sub_dir))

    sync_resp = business_client.post(f"/api/storages/localfiles/{storage['id']}/sync")
    assert sync_resp.status_code == 200, sync_resp.content
    tasks = list(Task.objects.filter(project_id=project_id))
    assert len(tasks) == 1
    assert '/data/local-files/?d=' in str(tasks[0].data)


@pytest.mark.django_db
def test_localfiles_keeps_absolute_path_compatible(business_client, settings, tmp_path):
    doc_root = tmp_path / 'files_root'
    sub_dir = doc_root / 'dataset1'
    sub_dir.mkdir(parents=True)

    settings.LOCAL_FILES_SERVING_ENABLED = True
    settings.LOCAL_FILES_DOCUMENT_ROOT = str(doc_root)

    project_id = _create_project(business_client)
    resp = business_client.post(
        '/api/storages/localfiles/',
        data=json.dumps({
            'project': project_id,
            'title': 'Absolute Source',
            'path': str(sub_dir),
            'use_blob_urls': True,
        }),
        content_type='application/json',
    )
    assert resp.status_code == 201, resp.content
    assert os.path.normpath(resp.json()['path']) == os.path.normpath(str(sub_dir))


@pytest.mark.django_db
def test_localfiles_blocks_relative_path_traversal_and_root(business_client, settings, tmp_path):
    doc_root = tmp_path / 'files_root'
    doc_root.mkdir(parents=True)
    outside_dir = tmp_path / 'outside_secret'
    outside_dir.mkdir(parents=True)

    settings.LOCAL_FILES_SERVING_ENABLED = True
    settings.LOCAL_FILES_DOCUMENT_ROOT = str(doc_root)

    project_id = _create_project(business_client)

    for bad_path in ('../outside_secret', '.'):
        resp = business_client.post(
            '/api/storages/localfiles/',
            data=json.dumps({
                'project': project_id,
                'title': 'Invalid Source',
                'path': bad_path,
                'use_blob_urls': True,
            }),
            content_type='application/json',
        )
        assert resp.status_code == 400, f'Expected 400 for {bad_path}, got {resp.status_code}: {resp.content}'


@pytest.mark.django_db
def test_localfiles_blocks_when_serving_disabled(business_client, settings, tmp_path):
    doc_root = tmp_path / 'files_root'
    sub_dir = doc_root / 'Raw' / 'images'
    sub_dir.mkdir(parents=True)

    settings.LOCAL_FILES_SERVING_ENABLED = False
    settings.LOCAL_FILES_DOCUMENT_ROOT = str(doc_root)

    project_id = _create_project(business_client)
    resp = business_client.post(
        '/api/storages/localfiles/',
        data=json.dumps({
            'project': project_id,
            'title': 'Disabled Source',
            'path': 'Raw/images',
            'use_blob_urls': True,
        }),
        content_type='application/json',
    )
    assert resp.status_code == 400, resp.content
