import datajoint as dj

reso = dj.create_virtual_module('reso','pipeline_reso')
meso = dj.create_virtual_module('meso','pipeline_meso')
stack = dj.create_virtual_module('stack','pipeline_stack')
shared = dj.create_virtual_module('shared','pipeline_shared')
experiment = dj.create_virtual_module('experiment','pipeline_experiment')
pupil = dj.create_virtual_module('pupil','pipeline_eye')
treadmill = dj.create_virtual_module('behavior','pipeline_treadmill')
stimulus = dj.create_virtual_module('stimulus','pipeline_stimulus')
virus = dj.create_virtual_module('virus','common_virus')
mice = dj.create_virtual_module('mice ','common_mice')