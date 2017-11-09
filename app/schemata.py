import datajoint as dj

reso = dj.create_virtual_module('reso','pipeline_reso')
meso = dj.create_virtual_module('meso','pipeline_meso')
shared = dj.create_virtual_module('shared','pipeline_shared')
experiment = dj.create_virtual_module('experiment','pipeline_experiment')
pupil = dj.create_virtual_module('pupil','pipeline_eye')
behavior = dj.create_virtual_module('behavior','pipeline_behavior')
stimulus = dj.create_virtual_module('stimulus','pipeline_stimulus')
vis = dj.create_virtual_module('vis','pipeline_vis')
virus = dj.create_virtual_module('virus','common_virus')
mice = dj.create_virtual_module('mice ','common_mice')