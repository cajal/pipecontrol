import datajoint as dj

reso = dj.create_virtual_module('reso','pipeline_reso')
meso = dj.create_virtual_module('meso','pipeline_meso')
stack = dj.create_virtual_module('stack','pipeline_stack')
shared = dj.create_virtual_module('shared','pipeline_shared')
experiment = dj.create_virtual_module('experiment','pipeline_experiment')
tune = dj.create_virtual_module('tune','pipeline_tune')
pupil = dj.create_virtual_module('pupil','pipeline_eye')
treadmill = dj.create_virtual_module('behavior','pipeline_treadmill')
stimulus = dj.create_virtual_module('stimulus','pipeline_stimulus')
xcorr = dj.create_virtual_module('xcorr','pipeline_xcorr')
mice = dj.create_virtual_module('mice ','common_mice')

dj.config['external-analysis'] = dict(
    protocol='file',
    location='/mnt/scratch05/datajoint-store/analysis')

dj.config['external-maps'] = dict(
    protocol='s3',
    endpoint="kobold.ad.bcm.edu:9000",
    bucket='microns-pipelines',
    location='maps',
    access_key="21IYGREPV4RS3IUU9ZYX",
    secret_key="yzGLiu7ndHzMSCrobTliCpRDpP9WGdRv7YmrieJ0")

# dj.config['cache'] = os.path.expanduser('/mnt/data/dj-cache')