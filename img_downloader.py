from planet import api
from planet.api import downloader

API_KEY = "d50e9f9afb324126aaf3b595ae51986d"

OUTDIR = '../img'

client = api.ClientV1()

dloader = downloader.create(client)

scene_ids = ['20170703_174536_0f52','20170707_181137_100b']

# Loop through ids
for scene_id in scene_ids:
    print('Downloading scene: ' + scene_id)

    # Get scene item
    item = client.get_item('PSScene3Band', scene_id).get()

    # Download visual asset to output directory
    r = dloader.download(iter([item]), ['visual'], OUTDIR)
