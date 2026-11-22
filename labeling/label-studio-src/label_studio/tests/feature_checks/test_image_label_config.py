import pytest
from django.urls import reverse


@pytest.mark.django_db
def test_accepts_image_polygon_label_config(business_client):
    label_config = """
    <View>
      <Image name="image" value="$image"/>
      <PolygonLabels name="labels" toName="image">
        <Label value="insect"/>
      </PolygonLabels>
    </View>
    """

    response = business_client.post(
        reverse("projects:api:label-config-validate"),
        data={"label_config": label_config},
        content_type="application/json",
    )

    assert response.status_code == 204, response.content