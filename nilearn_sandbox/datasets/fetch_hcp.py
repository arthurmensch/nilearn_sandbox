import glob
import os
from os.path import join

from nilearn.datasets.utils import _get_dataset_dir

from sklearn.datasets.base import Bunch


def fetch_hcp_rest(data_dir, n_subjects=40):
    dataset_name = 'HCP'
    source_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                  verbose=0)
    func = []
    meta = []
    list_dir = glob.glob(join(source_dir, '*/*/MNINonLinear/Results'))
    for dirpath in list_dir[:n_subjects]:
        dirpath_split = dirpath.split(os.sep)
        subject_id = dirpath_split[-3]
        serie_id = dirpath_split[-4]
        for filename in os.listdir(dirpath):
            name, ext = os.path.splitext(filename)
            if name in ('rfMRI_REST1_RL', 'rfMRI_REST1_LR',
                        'rfMRI_REST2_RL',
                        'rfMRI_REST2_LR'):
                filename = join(dirpath, filename, filename + '.nii.gz')
                func.append(filename)
                kwargs = {'record': name, 'subject_id': subject_id,
                          'serie_id': serie_id,
                          'filename': filename}
                meta.append(kwargs)
    results = {'func': func, 'meta': meta,
               'description': "'Human connectome project"}
    return Bunch(**results)


def fetch_hcp_reduced(data_dir, n_subjects=77):
    filenames = []
    for i in range(n_subjects):
        filenames.append(os.path.join(data_dir, 'HCP_reduced',
                                      'record_%i.nii.gz' % i))
    res = dict(func=filenames)
    return Bunch(**res)
