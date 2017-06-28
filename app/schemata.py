import datajoint as dj
# virus = dj.create_virtual_module('virus','common_virus')
# mice = dj.create_virtual_module('mice','common_mice')
reso = dj.create_virtual_module('reso','pipeline_reso')
experiment = dj.create_virtual_module('experiment','pipeline_experiment')
shared = dj.create_virtual_module('shared','pipeline_shared')
pupil = dj.create_virtual_module('shared','pipeline_pupil')
behavior = dj.create_virtual_module('behavior','pipeline_behavior')
meso = dj.create_virtual_module('meso','pipeline_meso')
